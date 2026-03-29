import yaml
from typing import Tuple

class Config():
    use_bfloat16: bool
    load_checkpoint_path: str
    checkpoint_type: str

    # -- context encoder
    encoder_model_name: str
    encoder_weights_source: str
    encoder_embed_dim: int

    # --- predictor
    predictor_model_name: str
    predictor_weights_source: str
    predictor_depth: int
    predictor_num_heads: int
    predictor_embed_dim: int

    # --- target encoder
    target_encoder_model_name: str
    target_encoder_weights_source: str

    # --- data
    batch_size: int
    crop_size: int
    patch_size: int
    crop_scale: Tuple[float, float]
    count_border_pixels: int

    image_folder: str
    data_root_folder_path: str
    pin_mem: bool
    num_workers: int

    # --- image transforms
    transform_mean: Tuple[float, float, float]
    transform_std: Tuple[float, float, float]

    # -- target blocks / masks
    target_block_scale: Tuple[float, float]
    target_block_aspect_ratio: Tuple[float, float]
    num_pred_masks: int
    allow_overlap: bool

    # --- context block / mask
    enc_mask_scale: Tuple[float, float]
    enc_mask_aspect_ratio: Tuple[float, float]
    enc_min_keep: int
    num_enc_masks: int

    # --- optimization
    num_epochs: int
    ema: Tuple[float, float]
    ipe_scale: float
    lr: float
    start_lr: float
    final_lr: float
    warmup: float
    wd: float
    final_wd: float
    
    # --- logging 
    logging_folder: str
    tag: str

def read_config(config_path: str) -> Config:
    config = Config()
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    config.use_bfloat16 = config_dict["meta"]["use_bfloat16"]
    config.load_checkpoint_path = config_dict["meta"]["load_checkpoint_path"]
    config.checkpoint_type = config_dict["meta"]["checkpoint_type"]

    assert config.checkpoint_type in ["dinov3", "ijepa"], "Checkpoint type should be: 'dinov3' or 'ijepa'"

    # --- context encoder
    config.encoder_model_name = config_dict["meta"]["encoder_model_name"]
    config.encoder_weights_source = config_dict["meta"]["encoder_weights_source"]
    config.encoder_embed_dim = config_dict["meta"]["encoder_embed_dim"]

    # --- target encoder
    config.target_encoder_model_name = config_dict["meta"]["target_encoder_model_name"]
    config.target_encoder_weights_source = config_dict["meta"]["target_encoder_weights_source"]

    # --- predictor
    config.predictor_model_name = config_dict["meta"]["predictor_model_name"]
    config.predictor_weights_source = config_dict["meta"]["predictor_weights_source"]
    config.predictor_depth = config_dict["meta"]["predictor_depth"]
    config.predictor_num_heads = config_dict["meta"]["predictor_num_heads"]
    config.predictor_embed_dim = config_dict["meta"]["predictor_embed_dim"]

    # --- data
    config.batch_size = config_dict["data"]["batch_size"]
    config.crop_size = config_dict["data"]["crop_size"]
    config.patch_size = config_dict["mask"]["patch_size"]
    config.crop_scale = config_dict["data"]["crop_scale"]
    config.count_border_pixels = config_dict["data"]["count_border_pixels"]

    config.image_folder = config_dict["data"]["image_folder"]
    config.data_root_folder_path = config_dict["data"]["data_root_folder_path"]
    config.pin_mem = config_dict["data"]["pin_mem"]
    config.num_workers = config_dict["data"]["num_workers"]
    
    # ---image transforms
    config.transform_mean = (
        config_dict["data"]["image_transform_statistics"]["mean"][0],
        config_dict["data"]["image_transform_statistics"]["mean"][1],
        config_dict["data"]["image_transform_statistics"]["mean"][2]
    )
    config.transform_std = (
    config_dict["data"]["image_transform_statistics"]["std"][0],
    config_dict["data"]["image_transform_statistics"]["std"][1],
    config_dict["data"]["image_transform_statistics"]["std"][2]
    )

    # -- target blocks / masks
    config.target_block_scale = config_dict["mask"]["pred_mask_scale"] # scale of target blocks
    config.target_block_aspect_ratio = config_dict["mask"]["pred_mask_aspect_ratio"] # aspect ratio of target blocks
    config.num_pred_masks = config_dict["mask"]["num_pred_masks"]  # number of target blocks
    config.allow_overlap = config_dict["mask"]["allow_overlap"]  # whether to allow overlap b/w context and target blocks
    
    # --- context block / mask
    config.enc_mask_scale = config_dict["mask"]["enc_mask_scale"] # scale of context blocks
    config.enc_mask_aspect_ratio = config_dict["mask"]["enc_mask_aspect_ratio"] # aspect ratio of the context blocks
    config.enc_min_keep = config_dict["mask"]["enc_min_keep"] # min number of patches in context block
    config.num_enc_masks = config_dict["mask"]["num_enc_masks"] # number of context blocks

    # --- optimization
    config.num_epochs = config_dict["optimization"]["num_epochs"]
    config.ema = config_dict["optimization"]["ema"]
    config.ipe_scale = config_dict["optimization"]["ipe_scale"]  # scheduler scale factor (def: 1.0)
    config.lr = config_dict["optimization"]["lr"]
    config.start_lr = config_dict["optimization"]["start_lr"]
    config.final_lr = config_dict["optimization"]["final_lr"]
    config.warmup = config_dict["optimization"]["warmup"]
    config.wd = config_dict["optimization"]["weight_decay"]
    config.final_wd = config_dict["optimization"]["final_weight_decay"]

    # -- logging
    config.logging_folder = config_dict["logging"]["logging_folder"]
    config.tag = config_dict["logging"]["write_tag"]

    return config

   