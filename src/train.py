# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import copy
import logging
import sys
import yaml

import numpy as np
from pathlib import Path

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.masks.multiblock import MaskCollator as MBMaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import (
    init_distributed,
    AllReduce
)
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter)
from src.utils.tensors import repeat_interleave_batch
from src.datasets.brain_mri import make_brain_mri

from src.helper import (
    load_models_from_checkpoint,
    init_model,
    init_opt,
    init_model_dinov3
)
from src.transforms import make_transforms

# --
log_timings = True
log_freq = 10
checkpoint_freq = 5
# --

_GLOBAL_SEED = 42
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True # maybe set this to False (default)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(args):
    # Setting the device
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = args['meta']['use_bfloat16']
    load_checkpoint_path = args['meta']['load_checkpoint_path']
    checkpoint_type = args['meta']['checkpoint_type']
    assert checkpoint_type in ['dinov3', 'ijepa'], "Checkpoint type should be: 'dinov3' or 'ijepa'"

    # context encoder
    encoder_model_name = args['meta']['encoder_model_name']
    encoder_weights_source = args['meta']['encoder_weights_source']
    encoder_embed_dim = args['meta']['encoder_embed_dim']

    # predictor
    predictor_model_name = args['meta']['predictor_model_name']
    predictor_weights_source = args['meta']['predictor_weights_source']
    predictor_depth = args['meta']['predictor_depth']
    predictor_embed_dim = args['meta']['predictor_embed_dim']
    predictor_num_heads = args['meta']['predictor_num_heads']

    # target encoder
    target_encoder_model_name = args['meta']['target_encoder_model_name']
    target_encoder_weights_source = args['meta']['target_encoder_weights_source']


    # -- DATA
    # training specifics
    batch_size = args['data']['batch_size']
    crop_size = args['data']['crop_size']
    crop_scale = args['data']['crop_scale']

    image_folder = args['data']['image_folder']
    data_root_folder_path = args['data']['data_root_folder_path']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']

    # data augmentation / transformations
    color_jitter_strength = args['data']['color_jitter_strength']
    use_color_distortion = args['data']['use_color_distortion']
    use_gaussian_blur = args['data']['use_gaussian_blur']
    use_horizontal_flip = args['data']['use_horizontal_flip']
    

    # -- MASK
    patch_size = args['mask']['patch_size']  # patch-size for model training
    allow_overlap = args['mask']['allow_overlap']  # whether to allow overlap b/w context and target blocks
    
    enc_mask_scale = args['mask']['enc_mask_scale']  # scale of context blocks
    enc_mask_aspect_ratio = args['mask']['enc_mask_aspect_ratio'] # aspect ratio of the context blocks
    enc_min_keep = args['mask']['enc_min_keep']  # min number of patches in context block
    num_enc_masks = args['mask']['num_enc_masks']  # number of context blocks
    
    pred_mask_scale = args['mask']['pred_mask_scale']  # scale of target blocks
    pred_mask_aspect_ratio = args['mask']['pred_mask_aspect_ratio']  # aspect ratio of target blocks
    num_pred_masks = args['mask']['num_pred_masks']  # number of target blocks
    

    # -- OPTIMIZATION
    num_epochs = args['optimization']['num_epochs']
    ema = args['optimization']['ema']
    ipe_scale = args['optimization']['ipe_scale']  # scheduler scale factor (def: 1.0)

    lr = args['optimization']['lr']
    start_lr = args['optimization']['start_lr']
    final_lr = args['optimization']['final_lr']
    warmup = args['optimization']['warmup']

    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    
    
    # -- LOGGING
    logging_folder = args['logging']['logging_folder']
    tag = args['logging']['write_tag']

    dump = os.path.join(logging_folder, 'config_run.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # -- log/checkpointing paths
    log_file = os.path.join(logging_folder, f'{tag}_r{rank}.csv')
    checkpoint_folder = logging_folder.replace("logs", "checkpoints")
    save_path = os.path.join(checkpoint_folder, f'{tag}' + '-epoch{epoch}.pth.tar')
    latest_path = os.path.join(checkpoint_folder, f'{tag}-latest.pth.tar')
    if load_checkpoint_path:
        load_path = Path(load_checkpoint_path)
    else:
        load_path = None

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%.5f', 'mask-A'),
                           ('%.5f', 'mask-B'),
                           ('%d', 'time (ms)'))

    # -- init model
    if checkpoint_type == "dinov3":
        dict_init_models = init_model_dinov3(
            device=device,
            encoder_model_name=encoder_model_name,
            encoder_weights_path=encoder_weights_source,
            predictor_depth=predictor_depth,
            encoder_embed_dim=encoder_embed_dim,
            predictor_embed_dim=predictor_embed_dim,
            predictor_num_patches=(crop_size // patch_size) * (crop_size // patch_size),
            predictor_num_heads=predictor_num_heads,
            target_encoder_model_name = target_encoder_model_name,
            target_encoder_weights_path = target_encoder_weights_source
        )
        encoder = dict_init_models["encoder"]
        predictor = dict_init_models["predictor"]
        target_encoder = dict_init_models["target_encoder"]

    elif checkpoint_type == "ijepa":
        encoder, predictor = init_model(
            device=device,
            patch_size=patch_size,
            crop_size=crop_size,
            pred_depth=predictor_depth,
            pred_emb_dim=predictor_embed_dim,
            model_name=encoder_model_name)
        target_encoder = copy.deepcopy(encoder)

    # -- make data transforms
    mask_collator = MBMaskCollator(
        input_size=crop_size,
        patch_size=patch_size,
        pred_mask_scale=pred_mask_scale,
        enc_mask_scale=enc_mask_scale,
        aspect_ratio=pred_mask_aspect_ratio,
        nenc=num_enc_masks,
        npred=num_pred_masks,
        allow_overlap=allow_overlap,
        min_keep=enc_min_keep)

    transform = make_transforms(
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=use_gaussian_blur,
        horizontal_flip=use_horizontal_flip,
        color_distortion=use_color_distortion,
        color_jitter=color_jitter_strength)

    # -- init data-loaders/samplers
    _, unsupervised_loader, unsupervised_sampler = make_brain_mri(
            transform=transform,
            batch_size=batch_size,
            collator=mask_collator,
            pin_mem=pin_mem,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=data_root_folder_path,
            image_folder=image_folder,
            drop_last=True
    )
    ipe = len(unsupervised_loader)
    print("Number of instances per epoch = ", ipe)

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16)

    # -- momentum schedule
    momentum_scheduler = (ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs * ipe_scale)
                          for i in range(int(ipe * num_epochs * ipe_scale) + 1))

    start_epoch = 0

    # -- load training checkpoint
    if load_path:
        loaded_dict_from_checkpoint = load_models_from_checkpoint(
            checkpoint_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            optimizer=optimizer,
            scaler=scaler,
            dinov3_ckpt=(checkpoint_type == "dinov3")
        )
        encoder = loaded_dict_from_checkpoint["encoder"]
        predictor = loaded_dict_from_checkpoint["predictor"]
        target_encoder = loaded_dict_from_checkpoint["target_encoder"]
        optimizer = loaded_dict_from_checkpoint["optimizer"]
        scaler = loaded_dict_from_checkpoint["scaler"]
        start_epoch = loaded_dict_from_checkpoint["epoch"] + 1

        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            mask_collator.step()

    encoder = DistributedDataParallel(encoder, static_graph=True)
    predictor = DistributedDataParallel(predictor, static_graph=True)
    target_encoder = DistributedDataParallel(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    def save_checkpoint(epoch):
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr
        }
        if checkpoint_type == "dinov3":
            save_dict["encoder_model_name"] = encoder_model_name
            save_dict["predictor_model_name"] = predictor_model_name
            save_dict["target_encoder_model_name"] = target_encoder_model_name
            save_dict["encoder_weights_source"] = encoder_weights_source
            save_dict["predictor_weights_source"] = predictor_weights_source
            save_dict["target_encoder_weights_source"] = target_encoder_weights_source

        if rank == 0:
            torch.save(save_dict, latest_path)
            if epoch % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=f'{epoch}'))

    # -- TRAINING LOOP
    print(f"Started training for {num_epochs} epochs...")
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        # -- update distributed-data-loader epoch
        unsupervised_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter = AverageMeter()

        for itr, (udata, masks_enc, masks_pred) in enumerate(unsupervised_loader):
            def load_imgs():
                # -- unsupervised imgs
                imgs = udata[0].to(device, non_blocking=True)
                masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
                masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]
                return (imgs, masks_1, masks_2)
            imgs, masks_enc, masks_pred = load_imgs()
            maskA_meter.update(len(masks_enc[0][0]))
            maskB_meter.update(len(masks_pred[0][0]))

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                def forward_target():
                    with torch.no_grad():
                        h = target_encoder.forward(imgs)
                        # We do not need this anymore as we're already applying a layer normalization at the end of the target encoder forward function
                        # h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
                        B = len(h)
                        # -- create targets (masked regions of h)
                        h = apply_masks(h, masks_pred)
                        h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
                        return h

                def forward_context():
                    z = encoder.forward(imgs, masks_enc)
                    z = predictor.forward(z, masks_enc, masks_pred)
                    return z

                def loss_fn(z, h):
                    loss = F.smooth_l1_loss(z, h)
                    loss = AllReduce.apply(loss)
                    return loss

                # Step 1. Forward
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    h = forward_target()
                    z = forward_context()
                    loss = loss_fn(z, h)

                #  Step 2. Backward & step
                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                grad_stats = grad_logger(encoder.named_parameters())
                optimizer.zero_grad()

                # Step 3. momentum update of target encoder
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

                return (float(loss), _new_lr, _new_wd, grad_stats)

            (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)

            # -- Logging
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, maskA_meter.val, maskB_meter.val, etime)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info('[%d, %5d] loss: %.3f '
                                'masks: %.1f %.1f '
                                '[wd: %.2e] [lr: %.2e] '
                                '[mem: %.2e] '
                                '(%.1f ms)'
                                % (epoch + 1, itr,
                                   loss_meter.avg,
                                   maskA_meter.avg,
                                   maskB_meter.avg,
                                   _new_wd,
                                   _new_lr,
                                   torch.cuda.max_memory_allocated() / 1024.**2,
                                   time_meter.avg))

                    if grad_stats is not None:
                        logger.info('[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                                    % (epoch + 1, itr,
                                       grad_stats.first_layer,
                                       grad_stats.last_layer,
                                       grad_stats.min,
                                       grad_stats.max))

            log_stats()

            assert not np.isnan(loss), 'loss is nan'
            torch.cuda.empty_cache()

        # -- Save Checkpoint after every epoch
        logger.info('avg. loss %.3f' % loss_meter.avg)
        save_checkpoint(epoch + 1)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
