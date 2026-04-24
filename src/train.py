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
from src.utils.configs import *

from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter)
from src.utils.tensors import repeat_interleave_batch
from src.datasets.brain_mri import make_brain_mri
from src.datasets.liver_ct import make_liver_ct
from src.datasets.resc import make_resc


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
checkpoint_freq = 10
# --

_GLOBAL_SEED = 42
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True # maybe set this to False (default)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(*args, **kwargs):
    # Setting the device
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # Read training config file
    config = read_train_ijepa_config(config_path=kwargs["config_path"])

    with open(kwargs["config_path"], "r") as f:
        yaml_config = yaml.safe_load(f)
    os.makedirs(config.logging_folder, exist_ok=False)
    dump = os.path.join(config.logging_folder, 'config_run.yaml')
    with open(dump, 'w') as f:
        yaml.dump(yaml_config, f)
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
    log_file = os.path.join(config.logging_folder, f'{config.tag}_r{rank}.csv')
    checkpoint_folder = config.logging_folder.replace("logs", "checkpoints")
    os.makedirs(checkpoint_folder, exist_ok=False)

    save_path = os.path.join(checkpoint_folder, f'{config.tag}' + '-epoch{epoch}.pth.tar')
    latest_path = os.path.join(checkpoint_folder, f'{config.tag}-latest.pth.tar')
    if config.load_checkpoint_path:
        load_path = Path(config.load_checkpoint_path)
    else:
        load_path = None

    dataset_name = config.image_folder.split("/")[0]

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%.5f', 'mask-A'),
                           ('%.5f', 'mask-B'),
                           ('%d', 'time (ms)'))

    # -- init model
    if config.checkpoint_type == "dinov3":
        dict_init_models = init_model_dinov3(
            device=device,
            encoder_model_name=config.encoder_model_name,
            encoder_weights_path=config.encoder_weights_source,
            predictor_depth=config.predictor_depth,
            encoder_embed_dim=config.encoder_embed_dim,
            predictor_embed_dim=config.predictor_embed_dim,
            predictor_num_patches=(config.crop_size // config.patch_size) * (config.crop_size // config.patch_size),
            predictor_num_heads=config.predictor_num_heads,
            target_encoder_model_name = config.target_encoder_model_name,
            target_encoder_weights_path = config.target_encoder_weights_source
        )
        encoder = dict_init_models["encoder"]
        predictor = dict_init_models["predictor"]
        target_encoder = dict_init_models["target_encoder"]

    elif config.checkpoint_type == "ijepa":
        encoder, predictor = init_model(
            device=device,
            patch_size=config.patch_size,
            crop_size=config.crop_size,
            pred_depth=config.predictor_depth,
            pred_emb_dim=config.predictor_embed_dim,
            model_name=config.encoder_model_name)
        target_encoder = copy.deepcopy(encoder)

    # -- make data transforms
    mask_collator = MBMaskCollator(
        input_size=config.crop_size,
        patch_size=config.patch_size,
        pred_mask_scale=config.target_block_scale,
        enc_mask_scale=config.enc_mask_scale,
        aspect_ratio=config.target_block_aspect_ratio,
        nenc=config.num_enc_masks,
        npred=config.num_pred_masks,
        allow_overlap=config.allow_overlap,
        min_keep=config.enc_min_keep)

    transform = make_transforms(
        crop_size=config.crop_size,
        crop_scale=config.crop_scale,
        normalization=(config.transform_mean, config.transform_std),
        dataset_name=dataset_name,
        count_border_pixels=config.count_border_pixels
    )

    # -- init data-loaders/samplers
    train_image_folder = os.path.join(config.image_folder, "train")
    valid_image_folder = os.path.join(config.image_folder, "valid/good/img")

    if "BraTS2021_slice" in config.image_folder:
        # Brain MRI dataset
        _, train_unsupervised_loader, train_unsupervised_sampler = make_brain_mri(
                transform=transform,
                batch_size=config.batch_size,
                collator=mask_collator,
                pin_mem=config.pin_mem,
                num_workers=config.num_workers,
                world_size=world_size,
                rank=rank,
                root_path=config.data_root_folder_path,
                image_folder=train_image_folder,
                drop_last=False,
                is_train=True,
        )

        _, valid_unsupervised_loader, _ = make_brain_mri(
                transform=transform,
                batch_size=config.batch_size,
                collator=mask_collator,
                pin_mem=config.pin_mem,
                num_workers=config.num_workers,
                world_size=world_size,
                rank=rank,
                root_path=config.data_root_folder_path,
                image_folder=valid_image_folder,
                drop_last=False,
                is_train=False,
        )
    elif "hist_DIY" in config.image_folder:
        # Liver CT dataset
        _, train_unsupervised_loader, train_unsupervised_sampler = make_liver_ct(
                transform=transform,
                batch_size=config.batch_size,
                collator=mask_collator,
                pin_mem=config.pin_mem,
                num_workers=config.num_workers,
                world_size=world_size,
                rank=rank,
                root_path=config.data_root_folder_path,
                image_folder=train_image_folder,
                drop_last=False,
                is_train=True,
        )

        _, valid_unsupervised_loader, _ = make_liver_ct(
                transform=transform,
                batch_size=config.batch_size,
                collator=mask_collator,
                pin_mem=config.pin_mem,
                num_workers=config.num_workers,
                world_size=world_size,
                rank=rank,
                root_path=config.data_root_folder_path,
                image_folder=valid_image_folder,
                drop_last=False,
                is_train=False,
        )
    elif "RESC" in config.image_folder:
        # RESC dataset
        _, train_unsupervised_loader, train_unsupervised_sampler = make_resc(
                transform=transform,
                batch_size=config.batch_size,
                collator=mask_collator,
                pin_mem=config.pin_mem,
                num_workers=config.num_workers,
                world_size=world_size,
                rank=rank,
                root_path=config.data_root_folder_path,
                image_folder=train_image_folder,
                drop_last=False,
                is_train=True,
        )

        _, valid_unsupervised_loader, _ = make_resc(
                transform=transform,
                batch_size=config.batch_size,
                collator=mask_collator,
                pin_mem=config.pin_mem,
                num_workers=config.num_workers,
                world_size=world_size,
                rank=rank,
                root_path=config.data_root_folder_path,
                image_folder=valid_image_folder,
                drop_last=False,
                is_train=False,
        )

    print("Train set = ", len(train_unsupervised_loader))
    print("Validation set = ", len(valid_unsupervised_loader))

    ipe = len(train_unsupervised_loader)
    print("Number of instances per epoch = ", ipe)

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=config.wd,
        final_wd=config.final_wd,
        start_lr=config.start_lr,
        ref_lr=config.lr,
        final_lr=config.final_lr,
        iterations_per_epoch=ipe,
        warmup=config.warmup,
        num_epochs=config.num_epochs,
        ipe_scale=config.ipe_scale,
        use_bfloat16=config.use_bfloat16)

    # -- momentum schedule
    momentum_scheduler = (config.ema[0] + i * (config.ema[1] - config.ema[0]) / (ipe * config.num_epochs * config.ipe_scale)
                          for i in range(int(ipe * config.num_epochs * config.ipe_scale) + 1))

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
            dinov3_ckpt=(config.checkpoint_type == "dinov3")
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
            'batch_size': config.batch_size,
            'world_size': world_size,
            'lr': config.lr
        }
        if config.checkpoint_type == "dinov3":
            save_dict["encoder_model_name"] = config.encoder_model_name
            save_dict["predictor_model_name"] = config.predictor_model_name
            save_dict["target_encoder_model_name"] = config.target_encoder_model_name
            save_dict["encoder_weights_source"] = config.encoder_weights_source
            save_dict["predictor_weights_source"] = config.predictor_weights_source
            save_dict["target_encoder_weights_source"] = config.target_encoder_weights_source

        if rank == 0:
            torch.save(save_dict, latest_path)

    # For early stopping
    best_valid_loss = float('inf')
    best_epoch = 0
    # -- TRAINING LOOP
    print(f"Started training for {config.num_epochs} epochs...")
    for epoch in range(start_epoch, config.num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        # -- update distributed-data-loader epoch
        train_unsupervised_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter = AverageMeter()

        for itr, (udata, masks_enc, masks_pred) in enumerate(train_unsupervised_loader):
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
                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=config.use_bfloat16):
                    h = forward_target()
                    z = forward_context()
                    loss = loss_fn(z, h)

                #  Step 2. Backward & step
                if config.use_bfloat16:
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

                return (loss.detach().item(), _new_lr, _new_wd, grad_stats)

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
            
            with torch.no_grad():
                log_stats()
                assert not np.isnan(loss), 'loss is nan'
                

        with torch.inference_mode():
            valid_losses = []

            for valid_itr, (valid_udata, valid_masks_enc, valid_masks_pred) in enumerate(valid_unsupervised_loader):
                valid_imgs = valid_udata.to(device, non_blocking=True)
                valid_masks_enc = [u.to(device, non_blocking=True) for u in valid_masks_enc]
                valid_masks_pred = [u.to(device, non_blocking=True) for u in valid_masks_pred]

                h_valid = target_encoder.forward(valid_imgs)
                # We do not need this anymore as we're already applying a layer normalization at the end of the target encoder forward function
                # h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
                B = len(h_valid)

                # -- create targets (masked regions of h)
                h_valid = apply_masks(h_valid, valid_masks_pred)
                h_valid = repeat_interleave_batch(h_valid, B, repeat=len(valid_masks_enc))

                z_valid = encoder.forward(valid_imgs, valid_masks_enc)
                z_valid = predictor.forward(z_valid, valid_masks_enc, valid_masks_pred)
                
                valid_loss = F.smooth_l1_loss(z_valid, h_valid)
                valid_loss = valid_loss.cpu().numpy()
                
                valid_losses.append(valid_loss)

        # -- Save Checkpoint after every epoch
        avg_valid_loss_epoch = np.mean(valid_losses)
        logger.info(f"avg. validation loss {avg_valid_loss_epoch}, epoch {epoch + 1}")
        if avg_valid_loss_epoch < best_valid_loss:
            best_valid_loss = avg_valid_loss_epoch
            best_epoch = epoch
            print(f"Saving the checkpoint at {latest_path}")
            save_checkpoint(epoch + 1)
        else:
            if epoch - best_epoch > 5: # TODO: make the patience configurable
                logger.info(f"The validation loss did not decrease in the last 5 epochs. Stopping the training.")
                break

        torch.cuda.empty_cache()

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
