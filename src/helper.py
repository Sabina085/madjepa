# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import sys
from typing import Dict, List, Optional

import torch
from torch import Tensor
from src.masks.utils import apply_masks

import src.models.vision_transformer as vit
from src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule)
from src.utils.tensors import trunc_normal_

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def convert_from_ddp_to_normal_state_dict(state_dict: Dict) -> Dict:
     # Distributed Data Parallel checkpoints are saved differently.
     # The names of the layers contain a "module." prefix. 
     # We use this function to remove this prefix
    state_dict_wo_prefix = {}
    for layer_name, weight in state_dict.items():
        layer_name_wo_prefix = layer_name.replace("module.", "")
        state_dict_wo_prefix[layer_name_wo_prefix] = weight

    return state_dict_wo_prefix


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.LayerNorm):
        torch.nn.init.constant_(m.bias, 0)
        torch.nn.init.constant_(m.weight, 1.0)


def load_checkpoint(
    r_path,
    encoder,
    predictor,
    target_encoder,
    opt,
    scaler,
    dinov3_ckpt=True
):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']

        # -- loading encoder
        encoder_pretrained_dict = checkpoint['encoder']
        encoder_pretrained_dict_not_ddp = convert_from_ddp_to_normal_state_dict(encoder_pretrained_dict)
        msg = encoder.load_state_dict(encoder_pretrained_dict_not_ddp)
        logger.info(f'Correctly loaded pretrained encoder from epoch {epoch} with msg: {msg}')
        if dinov3_ckpt:
            encoder.forward = encoder_forward.__get__(encoder)

        # -- loading predictor
        predictor_pretrained_dict = checkpoint['predictor']
        predictor_pretrained_dict_not_ddp = convert_from_ddp_to_normal_state_dict(predictor_pretrained_dict)
        msg = predictor.load_state_dict(predictor_pretrained_dict_not_ddp)
        logger.info(f'Correctly loaded pretrained predictor from epoch {epoch} with msg: {msg}')

        # -- loading target_encoder
        if target_encoder is not None:
            target_encoder_pretrained_dict = checkpoint['target_encoder']
            target_encoder_pretrained_dict_not_ddp = convert_from_ddp_to_normal_state_dict(target_encoder_pretrained_dict)
            msg = target_encoder.load_state_dict(target_encoder_pretrained_dict_not_ddp)
            logger.info(f'Correctly loaded pretrained target encoder from epoch {epoch} with msg: {msg}')

        if dinov3_ckpt:
            target_encoder.forward = target_encoder_forward.__get__(target_encoder)

        # -- loading optimizer
        if (opt and ('opt' in checkpoint) and checkpoint['opt']):
            opt.load_state_dict(checkpoint['opt'])
            logger.info(f'Loaded optimizer from epoch {epoch}')

        if (scaler and ('scaler' in checkpoint) and checkpoint['scaler']):
            scaler.load_state_dict(checkpoint['scaler'])
            logger.info(f'Loaded scaler from epoch {epoch}')
        
        logger.info(f'Read checkpoint from path: {r_path}')
        del checkpoint

    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')
        epoch = 0

    return encoder, predictor, target_encoder, opt, scaler, epoch


def init_model_dinov3(
        device: str,
        encoder_model_name: str,
        encoder_weights_path: str,
        predictor_num_patches: int,
        encoder_embed_dim: int,
        predictor_embed_dim: int,
        predictor_depth: int,
        predictor_num_heads: int,
        target_encoder_model_name: str,
        target_encoder_weights_path: str) -> Dict[str, object]:

    dinov3_github_location = "facebookresearch/dinov3"

    # -- Initializing encoder
    # Dinov3 VitS model
    encoder = torch.hub.load(
        repo_or_dir=dinov3_github_location,
        model=encoder_model_name,
        source="github",
        weights=encoder_weights_path
    )
    encoder.to(device=device)
    logger.info(f'Correctly initialized pretrained encoder (DinoV3 checkpoint)!')

    # -- Initializing predictor
    # Pre-trained IJepa checkpoint predictor model
    predictor = vit.__dict__['vit_predictor'](
        num_patches=predictor_num_patches,
        embed_dim=encoder_embed_dim,
        predictor_embed_dim=predictor_embed_dim,
        depth=predictor_depth,
        num_heads=predictor_num_heads
    )

    for m in predictor.modules():
        init_weights(m)

    predictor.to(device=device)
    logger.info(f'Correctly initialized pretrained predictor (IJepa checkpoint)!')

    # -- Initializing target_encoder
    target_encoder = torch.hub.load(
        repo_or_dir=dinov3_github_location,
        model=target_encoder_model_name,
        source="github",
        weights=target_encoder_weights_path
    )
    target_encoder.to(device=device)
    logger.info(f'Correctly initialized a pretrained target encoder (DinoV3 checkpoint)!')

    return {
        "encoder": encoder,
        "predictor": predictor,
        "target_encoder": target_encoder
    }


def init_model(
    device,
    patch_size=16,
    model_name='vit_base',
    crop_size=224,
    pred_depth=6,
    pred_emb_dim=384
):
    encoder = vit.__dict__[model_name](
        img_size=[crop_size],
        patch_size=patch_size)
    predictor = vit.__dict__['vit_predictor'](
        num_patches=encoder.patch_embed.num_patches,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
        num_heads=encoder.num_heads)

    for m in encoder.modules():
        init_weights(m)

    for m in predictor.modules():
        init_weights(m)

    encoder.to(device)
    predictor.to(device)
    logger.info(encoder)
    return encoder, predictor

def init_opt(
    encoder,
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_bfloat16=False,
    ipe_scale=1.25
):
    param_groups = [
        {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]

    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch))
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch))
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler


def encoder_forward(self: object, x: Tensor, masks: Optional[List[Tensor] | Tensor]) -> Tensor:
    if masks is not None:
        if not isinstance(masks, list):
            masks = [masks]

    x, (H, W) = self.prepare_tokens_with_masks(x=x, masks=None)
    rope_sincos = self.rope_embed(H=H, W=W)

    # -- mask x
    if masks is not None:
        x_special_tokens = x[:, :self.n_storage_tokens + 1, :]
        x_patch_tokens = x[:, self.n_storage_tokens + 1:, :]
        x_patch_tokens_masked = apply_masks(x_patch_tokens, masks)

        rope_sin = rope_sincos[0].unsqueeze(0)
        rope_cos = rope_sincos[1].unsqueeze(0)

        rope_sin_masked = apply_masks(rope_sin.expand(
            masks[0].shape[0], -1, -1), masks)[0]
        rope_cos_masked = apply_masks(rope_cos.expand(
            masks[0].shape[0], -1, -1), masks)[0]
        
        x = torch.concat([x_special_tokens, x_patch_tokens_masked], dim=1)
        rope_sincos = (rope_sin_masked, rope_cos_masked)

    for _, blk in enumerate(self.blocks):
        x = blk(x, rope_sincos)

    x_norm = self.norm(x)
    x_norm_patch = x_norm[:, self.n_storage_tokens + 1:, :]

    return x_norm_patch


def target_encoder_forward(self: object, x: Tensor) -> Tensor:
    x, (H, W) = self.prepare_tokens_with_masks(x=x, masks=None)
    rope_sincos = self.rope_embed(H=H, W=W)

    for _, blk in enumerate(self.blocks):
        x = blk(x, rope_sincos)

    x_norm = self.norm(x)
    x_norm_patch = x_norm[:, self.n_storage_tokens + 1:]

    return x_norm_patch
