import sys
sys.path.append("/home/vhondru/istoian/ijepa-anomaly-detection")

import torch
import logging
import src.models.vision_transformer as vit

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

CONFIG_CKPT = {
    "encoder_model_name": "dinov3_vits16",
    "encoder_weights_source": "pretrained_checkpoints/dinov3_models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "encoder_embed_dim": 384,
    "predictor_model_name": "ijepa_predictor_from_IN22K_vitg16_600",
    "ijepa_checkpoint_path": "pretrained_checkpoints/ijepa_checkpoints/IN22K-vit.g.16-600e.pth.tar",
    "predictor_num_patches": 196,
    "predictor_embed_dim": 384,
    "predictor_depth": 16,
    "predictor_num_heads": 16, # it seems that the number of heads for the predictor is set to be equal to the number of heads of the encoder (vitg in our case, as we are using the 16x16 I-Jepa checkpoint)
    "target_encoder_model_name": "dinov3_vits16",
    "target_encoder_weights_source": "pretrained_checkpoints/dinov3_models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "github_repo": "facebookresearch/dinov3",
    "save_path": "pretrained_checkpoints/init_checkpoints/enc_vits16_dinov3_pred_vitg16_ijepa_target_vits16_dinov3.pth.tar"
}

if __name__ == "__main__":
    # Use the DinoV3 checkpoint for the encoder
    encoder = torch.hub.load(
        repo_or_dir=CONFIG_CKPT["github_repo"],
        model=CONFIG_CKPT["encoder_model_name"],
        weights=CONFIG_CKPT["encoder_weights_source"],
        source="github"
    )
    print(f"Loaded pre-trained encoder - DinoV3 checkpoint for {CONFIG_CKPT['encoder_model_name']}")

    # Use the I-Jepa checkpoint for the predictor
    predictor = vit.__dict__['vit_predictor'](
        num_patches=CONFIG_CKPT["predictor_num_patches"],
        embed_dim=CONFIG_CKPT["encoder_embed_dim"],
        predictor_embed_dim=CONFIG_CKPT["predictor_embed_dim"],
        depth=CONFIG_CKPT["predictor_depth"],
        num_heads=CONFIG_CKPT["predictor_num_heads"])

    checkpoint = torch.load(CONFIG_CKPT["ijepa_checkpoint_path"], map_location=torch.device('cpu'))
    predictor_dict = checkpoint['predictor']
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
        predictor_dict, "module."
    )

    exclude_layers = [
        # these layers are the last two ones 
        # they map from the predictor embedding dimension to the target encoder embedding dimension
        # as the target encoder embedding size is different for us, we can not load these weights due to the shape mismatch
        "predictor_proj.weight", 
        "predictor_proj.bias",

        # this layer is among the first ones
        # it maps from the encoder embedding dimension to the predictor embedding dimesion
        # Our encoder has a different embedding dimension, hence we can not load these weights due to shape mismatch
        "predictor_embed.weight"
    ]

    filtered_predictor_dict = {}
    for layer_name, value in predictor_dict.items():
        if layer_name not in exclude_layers:
            filtered_predictor_dict[layer_name] = value

    msg = predictor.load_state_dict(filtered_predictor_dict, strict=False)
    logger.info(f'Correctly loaded the pretrained predictor from IJepa checkpoint.')

    # Use the DinoV3 checkpoint for the target_encoder
    target_encoder = torch.hub.load(
        repo_or_dir=CONFIG_CKPT["github_repo"],
        model=CONFIG_CKPT["target_encoder_model_name"],
        weights=CONFIG_CKPT["target_encoder_weights_source"],
        source="github"
    )
    print(f"Loaded pre-trained target encoder - DinoV3 checkpoint for"
          f" {CONFIG_CKPT['target_encoder_model_name']})")

    save_dict = {
        "encoder_model_name": CONFIG_CKPT["encoder_model_name"],
        "predictor_model_name": CONFIG_CKPT["predictor_model_name"],
        "target_encoder_model_name": CONFIG_CKPT["target_encoder_model_name"],
        'encoder': encoder.state_dict(),
        'predictor': predictor.state_dict(),
        'target_encoder': target_encoder.state_dict(),
        'opt': None,
        'scaler': None,
        'epoch': -1,
    }

    torch.save(save_dict, CONFIG_CKPT["save_path"])
