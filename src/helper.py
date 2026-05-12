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


def load_models_from_checkpoint(
    checkpoint_path: str,
    encoder: object,
    predictor: object,
    target_encoder: object,
    optimizer: Optional[object],
    scaler: Optional[object],
    dinov3_ckpt: Optional[bool] = True
) -> Dict[str, object]:
    """
    Function for loading the weights of the pre-trained encoder, predictor and target encoder from a checkpoint. 
    Optionally, the checkpoint could contain a saved optimizer and a scaler that will be also loaded. 
    
    :param checkpoint_path: str - checkpoint path
    :param encoder: object - the initialized encoder model object
    :param predictor: object - the initialized predictor model object
    :param target_encoder: object - the initialized target encoder model object
    :param optimizer: Optional[object] - the initialized optimizer object
    :param scaler: Optional[object] - the initialized scaler object
    :param dinov3_ckpt: bool = True - whether we will load a DinoV3 pretrained model (as the context and target encoders models) or not. Default:  True
    :return: Dict[str, object] - a dictionary containing the encoder, predictor, target_encoder, optimizer and the scaler 
    with the loaded weights and values from the checkpoint. The checkpoint witll also contain the epoch to be used for restarting the training.
    If it is the first training epoch, epoch = -1.
    """
    logger.info(f"Loading the checkpoint from the path {checkpoint_path}...\n")
    checkpoint = torch.load(f=checkpoint_path, map_location=torch.device('cpu'))
    epoch = checkpoint['epoch']

    # -- loading encoder
    encoder_pretrained_dict = checkpoint['encoder']
    encoder_pretrained_dict_not_ddp = convert_from_ddp_to_normal_state_dict(encoder_pretrained_dict)
    msg = encoder.load_state_dict(encoder_pretrained_dict_not_ddp, strict=True)
    if dinov3_ckpt:
        # Override the forward function of the context encoder model with a custom forward function
        # Necessary when loading a pre-trained DinoV3 model to be used as the (context) encoder
        encoder.forward = encoder_forward_dinov3.__get__(encoder)
    logger.info(f'Correctly loaded pretrained encoder from epoch {epoch} with msg: {msg}')

    # -- loading predictor
    predictor_pretrained_dict = checkpoint['predictor']
    predictor_pretrained_dict_not_ddp = convert_from_ddp_to_normal_state_dict(predictor_pretrained_dict)
    msg = predictor.load_state_dict(predictor_pretrained_dict_not_ddp, strict=True)
    logger.info(f'Correctly loaded pretrained predictor from epoch {epoch} with msg: {msg}')

    # -- loading target_encoder
    target_encoder_pretrained_dict = checkpoint['target_encoder']
    target_encoder_pretrained_dict_not_ddp = convert_from_ddp_to_normal_state_dict(target_encoder_pretrained_dict)
    msg = target_encoder.load_state_dict(target_encoder_pretrained_dict_not_ddp, strict=True)
    if dinov3_ckpt:
        # Override the forward function of the target encoder model with a custom forward function
        # Necessary when loading a pre-trained DinoV3 model to be used as the target encoder
        target_encoder.forward = target_encoder_forward_dinov3.__get__(target_encoder)
    logger.info(f'Correctly loaded pretrained target encoder from epoch {epoch} with msg: {msg}')

    # -- loading the optimizer if it exists and it is not null
    if (optimizer and ('opt' in checkpoint) and checkpoint['opt']):
        optimizer.load_state_dict(checkpoint['opt'])
        logger.info(f'Loaded optimizer from epoch {epoch}')

    # -- loading the scaler if it exists and it is not null
    if (scaler and ('scaler' in checkpoint) and checkpoint['scaler']):
        scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f'Loaded scaler from the checkpoint. Epoch {epoch}')
    
    logger.info(f'Correctly loaded the weights of the models from the checkpoint saved at the path: {checkpoint_path}')

    return {
        "encoder": encoder,
        "predictor": predictor,
        "target_encoder": target_encoder,
        "optimizer": optimizer,
        "scaler": scaler,
        "epoch": epoch
    }

# Set dropout on all existing Dropout modules
def set_dropout(model, drop_rate=0.1):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout) and "mlp" in name:
            module.p = drop_rate

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
        target_encoder_weights_path: str,
        encoder_mlp_dropout: float = 0.0,
        predictor_mlp_dropout: float = 0.0
    ) -> Dict[str, object]:
    
    
    assert encoder_model_name == target_encoder_model_name, "Both the encoder and the target encoder should have the same architecture"
    assert encoder_weights_path == target_encoder_weights_path, "Both the encoder and the target encoder should be initialized with identical weights"

    dinov3_github_location = "facebookresearch/dinov3"

    # -- Initializing encoder
    # Dinov3 model
    encoder = torch.hub.load(
        repo_or_dir=dinov3_github_location,
        model=encoder_model_name,
        source="github",
        weights=encoder_weights_path
    )
    encoder.to(device=device)

    set_dropout(model=encoder, drop_rate=encoder_mlp_dropout)
    logger.info(f'Correctly initialized pretrained encoder (DinoV3 checkpoint)!')

    # -- Initializing predictor
    # Will be used for loading a pre-trained IJepa predictor 
    predictor = vit.__dict__['vit_predictor'](
        num_patches=predictor_num_patches,
        embed_dim=encoder_embed_dim, # embedding dimension of the encoder
        predictor_embed_dim=predictor_embed_dim, # embedding dimension of the predictor
        depth=predictor_depth,
        num_heads=predictor_num_heads,
        drop_rate=predictor_mlp_dropout
    )
    for m in predictor.modules():
        init_weights(m)
    predictor.to(device=device)
    logger.info(f'Correctly initialized pretrained predictor (to load the weights from an IJepa checkpoint)!')

    # -- Initializing target encoder
    # Dinov3 model - the same architecture and the same checkpoint used as the encoder model
    target_encoder = torch.hub.load(
        repo_or_dir=dinov3_github_location,
        model=target_encoder_model_name, # should be the same as encoder_model_name
        source="github",
        weights=target_encoder_weights_path # should be the same as encoder_weights_path
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
    scaler = torch.amp.GradScaler('cuda') if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler


def encoder_forward_dinov3(self: object, x: Tensor, masks: Optional[List[Tensor] | Tensor]) -> Tensor:
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
    x_norm_patch = x_norm[:, self.n_storage_tokens + 1 :, :]

    return x_norm_patch


def target_encoder_forward_dinov3(self: object, x: Tensor) -> Tensor:
    # Used DinoV3 implementation: https://github.com/facebookresearch/dinov3/blob/main/dinov3/models/vision_transformer.py#L222
    x, (H, W) = self.prepare_tokens_with_masks(x=x, masks=None)
    # H, W -> computed after patch embeddings
    # Rotary Positional Embeddings (RoPE) used by DinoV3 models
    rope_sincos = self.rope_embed(H=H, W=W)

    for _, blk in enumerate(self.blocks):
        x = blk(x, rope_sincos)

    x_norm = self.norm(x)

    # [CLS] token -> position 0; 
    # DinoV3 uses n_storage_tokens on sequence positions [1, n_storage_tokens] 
    # Patch tokens will be from n_storage_tokens + 1 index to the end of the sequence
    x_norm_patch = x_norm[:, self.n_storage_tokens + 1 :, :]

    return x_norm_patch
