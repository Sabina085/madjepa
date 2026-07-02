"""
Linear classifier on BMAD synthetic anomaly dataset using a frozen I-JEPA target encoder.

The train split provides (normal, synthetic_anomaly, anomaly_map) triplets from:
  - acroitoru/bbmad_restructured —> normal images
  - acroitoru/medical_anomalies_qwen —> paired synthetic anomalies

Each triplet yields two labelled samples: normal (label=0) and synthetic anomaly (label=1).
The val/test splits have real labels from the BMAD benchmark.
"""

import argparse
import logging
import os
import sys
import glob
import hashlib
import json
import shutil

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

import datasets as hf_datasets
import torch.nn.functional as F

from src.helper import convert_from_ddp_to_normal_state_dict, target_encoder_forward_dinov3
from src.transforms import CropBackgroundBordersTransform
from src.utils.configs import read_train_ijepa_config, read_train_synthetic_anomaly_classifier_config
from instance_specific_config import ROOT_DIRECTORY_EXPERIMENTS_PATH

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


class _BMADBase:
    # Used for filtering the hugging face dataset by subset name in original_path
    def _load_and_filter(self, hf_split: str, subset: str):
        ds = hf_datasets.load_dataset("acroitoru/bbmad_restructured")[hf_split]
        indices = [i for i, p in enumerate(ds["original_path"]) if subset in p]

        return ds.select(indices), indices


class TrainClassificationDataset(_BMADBase, Dataset):
    """
    Flat (image, label) dataset built from BMAD train triplets.

    Each underlying triplet (normal, synthetic_anomaly, anomaly_map) produces two
    samples — the normal image (label 0) followed by the synthetic anomaly (label 1) —
    so __len__ returns 2 * len(triplets).
    """

    def __init__(self, subset: str, transform):
        self.transform = transform
        self.normal_ds, indices = self._load_and_filter("train", subset)
        self.anomaly_ds = hf_datasets.load_dataset("acroitoru/medical_anomalies_qwen")["train"].select(indices)

        assert len(self.normal_ds) == len(self.anomaly_ds)

    def __len__(self):
        return len(self.normal_ds) * 2

    def __getitem__(self, idx):
        base_idx = idx // 2
        is_anomaly = idx % 2
        if is_anomaly:
            img = self.anomaly_ds[base_idx]["anomaly"].convert("RGB")
        else:
            img = self.normal_ds[base_idx]["image"].convert("RGB")

        return self.transform(img), float(is_anomaly)


class EvalClassificationDataset(_BMADBase, Dataset):
    def __init__(self, split: str, subset: str, transform):
        self.transform = transform
        hf_split = "validation" if split == "val" else split
        self.ds, _ = self._load_and_filter(hf_split, subset)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        img = sample["image"].convert("RGB")
        label = float(sample["label"] != "good")

        return self.transform(img), label


class HeadClassifier(nn.Module):
    """Linear head on top of mean-pooled patch embeddings."""
    def __init__(self, embed_dim: int, hidden_size: int, activation: str, drop_rate: float=0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_size)
        if activation == "relu":
            self.activ1 = nn.ReLU()
        elif activation == "tanh":
            self.activ1 = nn.Tanh()
        elif activation == "selu":
            self.activ1 = nn.SELU()
        elif activation == "gelu":
            self.activ1 = nn.GELU()

        self.dropout = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(hidden_size, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:        
        x1 = self.activ1(self.fc1(x))
        x1 = self.dropout(x1)
        x2 = self.fc2(x1)

        return x2.squeeze(1)


@torch.no_grad()
def extract_features(encoder, images: torch.Tensor, layer_norm_teacher) -> torch.Tensor:
    # Run frozen target encoder and mean-pool patch tokens → [B, D]
    patch_tokens = encoder(images)   # [B, N, D]

    if layer_norm_teacher:
        patch_tokens = F.layer_norm(patch_tokens, (patch_tokens.size(-1),))  # normalize over feature-dim

    return patch_tokens.mean(dim=1)  # [B, D]


@torch.no_grad()
def evaluate(encoder, classifier, loader, device, layer_norm_teacher) -> dict:
    classifier.eval()
    all_logits, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        features = extract_features(
            encoder, 
            images, 
            layer_norm_teacher=layer_norm_teacher
        )
        logits = classifier(features)
        all_logits.append(logits.cpu())
        all_labels.append(labels)

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy().astype(int)

    probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
    preds = (probs >= 0.5).astype(int)
    auroc = roc_auc_score(all_labels, probs) if len(np.unique(all_labels)) > 1 else float("nan")
    bacc = balanced_accuracy_score(all_labels, preds)

    return {"auroc": auroc, "balanced_acc": bacc}


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    classifer_config = read_train_synthetic_anomaly_classifier_config(args.classifier_config_path)
    experiment_folder_path = os.path.join(ROOT_DIRECTORY_EXPERIMENTS_PATH, classifer_config.experiment_name)

    # read I-JEPA config for model architecture and image normalisation
    ijepa_train_config_path = os.path.join(experiment_folder_path, "logs/config_run.yaml")
    ijepa_train_config = read_train_ijepa_config(ijepa_train_config_path)

    # image transforms (same stats as I-JEPA training)
    transforms_list = []
    if classifer_config.subset in ["hist_DIY"]:
        transforms_list.append(
            CropBackgroundBordersTransform(classifer_config.subset, ijepa_train_config.count_border_pixels))
        
    transforms_list += [
        transforms.Resize((ijepa_train_config.crop_size, ijepa_train_config.crop_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=ijepa_train_config.transform_mean, std=ijepa_train_config.transform_std),
    ]
    image_transform = transforms.Compose(transforms_list)

    # -- datasets & loaders
    logger.info(f"Loading datasets for subset: {classifer_config.subset}")
    train_ds = TrainClassificationDataset(subset=classifer_config.subset, transform=image_transform)
    val_ds = EvalClassificationDataset(split="val", subset=classifer_config.subset, transform=image_transform)

    train_loader = DataLoader(train_ds, batch_size=classifer_config.batch_size, shuffle=True,
                              num_workers=classifer_config.num_workers, pin_memory=True)
    val_loader = DataLoader(
        val_ds, 
        batch_size=classifer_config.batch_size, 
        shuffle=False,
        num_workers=classifer_config.num_workers, pin_memory=True
    )

    logger.info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # load frozen target encoder
    logger.info("Loading target encoder from checkpoint...")
    target_encoder = torch.hub.load(
        "facebookresearch/dinov3",
        ijepa_train_config.target_encoder_model_name,
        source="github",
        weights=ijepa_train_config.target_encoder_weights_source,
    )
    target_encoder.forward = target_encoder_forward_dinov3.__get__(target_encoder)

    checkpoint_path = glob.glob(os.path.join(experiment_folder_path, "checkpoints/*-latest.pth.tar"))[0]
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = convert_from_ddp_to_normal_state_dict(checkpoint["target_encoder"])
    target_encoder.load_state_dict(state_dict)
    for p in target_encoder.parameters():
        p.requires_grad = False
    target_encoder.eval().to(device)
    logger.info(f"Target encoder loaded and frozen from checkpoint: {checkpoint_path}")

    # head classifier
    classifier = HeadClassifier(
        embed_dim=ijepa_train_config.encoder_embed_dim, 
        hidden_size=classifer_config.hidden_size, 
        activation=classifer_config.activation,
        drop_rate=classifer_config.drop_rate
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=classifer_config.lr,
        weight_decay=classifer_config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=classifer_config.scheduler_factor,
        patience=classifer_config.scheduler_patience,
        min_lr=classifer_config.scheduler_min_lr,
    )
    criterion = nn.BCEWithLogitsLoss()

    # Set the saving paths for the trained checkpoints
    classifier_save_path = os.path.join(experiment_folder_path, "checkpoints_classifier")

    # Create a hash based on the training config of the classifcation model
    json_classifier_config_sorted = json.dumps(vars(classifer_config), sort_keys=True)
    classification_model_hash = hashlib.sha256(json.dumps(json_classifier_config_sorted).encode()).hexdigest()[: 16]
    classifier_save_path = os.path.join(classifier_save_path, classification_model_hash)
    os.makedirs(classifier_save_path, exist_ok=False)

    best_auroc = -1.0
    best_epoch = -1

    for epoch in range(classifer_config.num_epochs):
        classifier.train()
        total_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{classifer_config.num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            features = extract_features(
                target_encoder, 
                images, 
                layer_norm_teacher=ijepa_train_config.layer_norm_teacher
            )
            logits = classifier(features)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(images)

        avg_loss = total_loss / len(train_ds)
        val_metrics = evaluate(
            target_encoder, 
            classifier, 
            val_loader, 
            device, 
            layer_norm_teacher=ijepa_train_config.layer_norm_teacher
        )
        scheduler.step(val_metrics["auroc"])

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch + 1:3d} | loss={avg_loss:.4f} | "
            f"val AUROC={val_metrics['auroc']:.4f} | "
            f"val balanced_acc={val_metrics['balanced_acc']:.4f} | "
            f"lr={current_lr:.2e}"
        )

        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            best_epoch = epoch + 1
            # We remove the previously saved checkpoint
            classifier_existing_path = glob.glob(os.path.join(classifier_save_path, f"{classification_model_hash}-classifier-latest-*.pth"))
            if len(classifier_existing_path) > 0:
                os.remove(classifier_existing_path[0])
            torch.save(
                {"epoch": best_epoch, "classifier_state_dict": classifier.state_dict(), "val_metrics": val_metrics},
                os.path.join(classifier_save_path, f"{classification_model_hash}-classifier-latest-{round(val_metrics['auroc'] * 100, 2)}.pth"),
            )
            shutil.copy2(args.classifier_config_path, os.path.join(classifier_save_path, "config.yaml"))
            logger.info(f"  -> New best saved (AUROC={best_auroc:.4f}) to {os.path.join(classifier_save_path, f"{classification_model_hash}-classifier-latest-{round(val_metrics['auroc'] * 100, 2)}.pth")}")

    logger.info(f"Training done. Best val AUROC={best_auroc:.4f} at epoch {best_epoch}.")
    

def parse_args():
    parser = argparse.ArgumentParser(description="Train a linear classifier on BMAD synthetic anomaly dataset")
    parser.add_argument("--classifier_config_path", type=str, default="BraTS",
                        help="Train classifier config path")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
