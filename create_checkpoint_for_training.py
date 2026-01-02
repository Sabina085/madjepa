import torch
import logging
import sys
import src.models.vision_transformer as vit

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

CONFIG_CKPT = {
    "encoder_model_name": "dinov3_vits16",
    "encoder_weights_path": "pretrained_checkpoints/dinov3_models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "encoder_embed_dim": 384,
    "predictor_model_name": "vit_tiny_ijepa",
    "ijepa_checkpoint_path": "pretrained_checkpoints/ijepa_checkpoints/IN22K-vit.h.14-900e.pth.tar",
    "predictor_num_patches": 196,
    "predictor_embed_dim": 384,
    "predictor_depth": 12,
    "predictor_num_heads": 6,
    "target_encoder_model_name": "dinov3_vits16",
    "target_encoder_weights_path": "pretrained_checkpoints/dinov3_models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "github_repo": "facebookresearch/dinov3",
    "save_path": "pretrained_checkpoints/dinov3_checkpoints/dinov3_enc_vits16_pred_vit_tiny_ijepa_target_vits16.pth.tar"
}

if __name__ == "__main__":
    # Use the DinoV3 checkpoint for the encoder
    encoder = torch.hub.load(
        repo_or_dir=CONFIG_CKPT["github_repo"],
        model=CONFIG_CKPT["encoder_model_name"],
        weights=CONFIG_CKPT["encoder_weights_path"],
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
        "predictor_proj.weight",
        "predictor_proj.bias",
    #     "predictor_embed.weight",
    #     "predictor_pos_embed"
    ]

    exclude_layers = []

    filtered_predictor_dict = {}
    for layer_name, value in predictor_dict.items():
        if layer_name not in exclude_layers:
            filtered_predictor_dict[layer_name] = value

    msg = predictor.load_state_dict(filtered_predictor_dict, strict=True)
    logger.info(f'Correctly loaded the pretrained predictor from IJepa checkpoint.')

    # Use the DinoV3 checkpoint for the target_encoder
    target_encoder = torch.hub.load(
        repo_or_dir=CONFIG_CKPT["github_repo"],
        model=CONFIG_CKPT["target_encoder_model_name"],
        weights=CONFIG_CKPT["target_encoder_weights_path"],
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
