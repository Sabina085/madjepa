import torch
import argparse
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import seaborn as sns
import pandas as pd
import shutil

from torchvision import transforms
from src.utils.configs import *
from src.utils.inference import *
from src.utils.dataset import *
from src.helper import (
    load_models_from_checkpoint,
    init_model_dinov3
)
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import torch.nn.functional as F
from transformers import ViTConfig, ViTModel

# STORAGE_PARENT_FOLDER_PATH = "/home/sabina/PhD/BMAD"
STORAGE_PARENT_FOLDER_PATH = "/media/vhondru/hdd/istoian/BMAD/dataset"

class DistilledModelTrainer:
    def __init__(self, train_distillation_config_path: str):

        self.train_distillation_config_path = train_distillation_config_path
        # Read the configuration file which will be used for training the student model to mimic the target encoder
        self.train_distillation_config = read_train_distillation_config(config_path=self.train_distillation_config_path)

        self._setup_device()
        self._set_prerequisites()
        self._init_student_model()
        self._init_target_encoder()
        self._set_optimization_configuration()

    def _set_prerequisites(self):
        subpath_list = os.getcwd().split(os.sep)
        root_directory_path = os.path.join(*subpath_list[:-1])
        current_dir = subpath_list[-1]

        # self.experiment_folder_path = (
        #     f"/{root_directory_path}/{current_dir}/"
        #     f"experiments_{self.train_distillation_config.dataset_name}/"
        #     f"{self.train_distillation_config.experiment_name}"
        # )

        self.experiment_folder_path = (
            f"/media/vhondru/hdd/istoian/ijepa-anomaly-detection/"
            f"experiments_{self.train_distillation_config.dataset_name}/"
            f"{self.train_distillation_config.experiment_name}"
        )

        self.ijepa_ckpt_folder_path = (
            f"{self.experiment_folder_path}/checkpoints"
        )

        self.train_ijepa_config_path = (
            f"{self.experiment_folder_path}/logs/config_run.yaml"
        )

        # Read the configuration file (which was used for training the target encoder)
        self.train_ijepa_config = read_train_ijepa_config(config_path=self.train_ijepa_config_path)

        self.ijepa_ckpt_file_path = f"{self.ijepa_ckpt_folder_path}/{self.train_ijepa_config.tag}-latest.pth.tar"

        # Create a hash based on the training config of the distilled model
        json_train_distillation_config_sorted = json.dumps(vars(self.train_distillation_config), sort_keys=True)
        self.distillation_config_hash = hashlib.sha256(json.dumps(json_train_distillation_config_sorted).encode()).hexdigest()[: 16]

        # self.distilled_model_folder_path = (
        #     f"/{root_directory_path}/{current_dir}/"
        #     f"experiments_{self.train_distillation_config.dataset_name}/"
        #     f"{self.train_distillation_config.experiment_name}/checkpoints_distilled_model/{self.distillation_config_hash}"
        # )

        self.distilled_model_folder_path = (
            f"/media/vhondru/hdd/istoian/ijepa-anomaly-detection/"
            f"experiments_{self.train_distillation_config.dataset_name}/"
            f"{self.train_distillation_config.experiment_name}/checkpoints_distilled_model/{self.distillation_config_hash}"
        )

        os.makedirs(self.distilled_model_folder_path, exist_ok=False)

    def _load_dataset(self):
        train_dataset_path = f"{STORAGE_PARENT_FOLDER_PATH}/{self.train_distillation_config.dataset_name}/train"
        valid_dataset_path = f"{STORAGE_PARENT_FOLDER_PATH}/{self.train_distillation_config.dataset_name}/valid"

        # Load the validation dataset
        image_transform = transforms.Compose([
            transforms.Resize((self.train_ijepa_config.crop_size, self.train_ijepa_config.crop_size), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(
                    self.train_ijepa_config.transform_mean[0],
                    self.train_ijepa_config.transform_mean[1],
                    self.train_ijepa_config.transform_mean[2]
                ),
                std=(
                    self.train_ijepa_config.transform_std[0],
                    self.train_ijepa_config.transform_std[1],
                    self.train_ijepa_config.transform_std[2]
                )
            ),
        ])

        self.train_dataset = ImageDataset(
            images_dir=train_dataset_path,
            dataset_name=self.train_distillation_config.dataset_name,
            count_border_pixels=self.train_ijepa_config.count_border_pixels,
            transform=image_transform
        )

        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_distillation_config.train_batch_size,
            shuffle=True,
            drop_last=False
        )

        self.valid_dataset = LabelDataset(
            images_dir=valid_dataset_path,
            dataset_name=self.train_distillation_config.dataset_name,
            count_border_pixels=self.train_ijepa_config.count_border_pixels,
            transform=image_transform
        )

        self.valid_loader = torch.utils.data.DataLoader(
            dataset=self.valid_dataset,
            batch_size=32,
            shuffle=False,
            drop_last=False
        )

        print("Number of training samples: ", len(self.train_dataset))
        print("Number of training batches: ", len(self.train_loader))
        print()
        print("Number of validation samples: ", len(self.valid_dataset))
        print("Number of validation batches: ", len(self.valid_loader))

    def _setup_device(self):
        # Set up the device
        if not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:0')
            torch.cuda.set_device(self.device)

        print(f"Using device: {self.device}")

    def _init_student_model(self):
        # Initializing a ViT model with a custom configuration
        # 6 transformer blocks, 6 attention heads, 384 hidden size, mlp_ratio=4 
        configuration = ViTConfig(
            hidden_size=self.train_distillation_config.hidden_size,
            num_hidden_layers=self.train_distillation_config.num_hidden_layers,
            num_attention_heads=self.train_distillation_config.num_attention_heads,
            intermediate_size=self.train_distillation_config.intermediate_size
        )

        # Initializing a VIT model with random weights
        self.student_model = ViTModel(configuration).to(self.device)
    
    def _init_target_encoder(self):
        # Init DinoV3/I-Jepa models: (context) encoder (DinoV3) + predictor (I-Jepa checkpoint) + target_encoder (DinoV3 - same architecture as the context encoder)
        dict_models = init_model_dinov3(
            device=self.device,
            encoder_model_name=self.train_ijepa_config.encoder_model_name,
            encoder_weights_path=self.train_ijepa_config.encoder_weights_source,
            predictor_depth=self.train_ijepa_config.predictor_depth,
            encoder_embed_dim=self.train_ijepa_config.encoder_embed_dim,
            predictor_embed_dim=self.train_ijepa_config.predictor_embed_dim,
            predictor_num_patches=(self.train_ijepa_config.crop_size // self.train_ijepa_config.patch_size) * (self.train_ijepa_config.crop_size // self.train_ijepa_config.patch_size),
            predictor_num_heads=self.train_ijepa_config.predictor_num_heads,
            target_encoder_model_name=self.train_ijepa_config.target_encoder_model_name,
            target_encoder_weights_path=self.train_ijepa_config.target_encoder_weights_source
        )

        # Load the pre-trained checkpoint (encoder + target encoder - DinoV3 VitS, predictor - IJepa checkpoint)
        print(f"Loading ijepa checkpoint from: {self.ijepa_ckpt_file_path}")
        loaded_dict_from_checkpoint = load_models_from_checkpoint(
            checkpoint_path=self.ijepa_ckpt_file_path,
            encoder=dict_models["encoder"],
            predictor=dict_models["predictor"],
            target_encoder=dict_models["target_encoder"],
            optimizer=None,
            scaler=None,
            dinov3_ckpt=True
        )

        self.target_encoder = loaded_dict_from_checkpoint["target_encoder"]

    def _loss_fn(self, student, teacher):
        # student, teacher: [B, N_patches, 384]
        assert student.shape == teacher.shape

        student_norm = F.normalize(student, dim=-1)  # normalize each patch vector independently
        teacher_norm = F.normalize(teacher, dim=-1)
        
        return self.criterion(student_norm, teacher_norm)
    
    def _loss_fn_valid(self, student, teacher):
        # student, teacher: [B, N_patches, 384]
        student_norm = F.normalize(student, dim=-1)  # normalize each patch vector independently
        teacher_norm = F.normalize(teacher, dim=-1)

        return self.valid_criterion(student_norm, teacher_norm)

    def _set_optimization_configuration(self):
        # Freeze the target encoder weights
        for name, param in self.target_encoder.named_parameters():
            param.requires_grad = False

        # Feed the trainable parameters of the student to the optimizer
        trainable_params = [param for param in self.student_model.parameters() if param.requires_grad]
        self.optimizer = torch.optim.AdamW(
            params=trainable_params,
            lr=self.train_distillation_config.peak_lr,
            weight_decay=self.train_distillation_config.weight_decay
        )
        self.criterion = torch.nn.MSELoss()
        self.valid_criterion = torch.nn.MSELoss(reduction="none")
        
        # self.criterion = torch.nn.SmoothL1Loss()
        # self.valid_criterion = torch.nn.SmoothL1Loss(reduction="none")

        # Learning rate schedulers
        # Phase 1: Linear Warmup -> from START_LR up to PEAK_LR
        self.warmup_sched = LinearLR(
            self.optimizer, 
            start_factor=self.train_distillation_config.start_lr / self.train_distillation_config.peak_lr, 
            total_iters=self.train_distillation_config.warmup_epochs
        )

        # Phase 2: Cosine Annealing -> from 1e-3 (PEAK_LR) down to 1e-5 (FINAL_LR)
        self.cosine_sched = CosineAnnealingLR(
            self.optimizer, 
            T_max=self.train_distillation_config.num_epochs - self.train_distillation_config.warmup_epochs, 
            eta_min=self.train_distillation_config.final_lr
        )

        self.scheduler = SequentialLR(
            self.optimizer, 
            schedulers=[self.warmup_sched, self.cosine_sched], 
            milestones=[self.train_distillation_config.warmup_epochs]
        )

    def _save_loss_plot(
            self, 
            num_epochs: int, 
            train_loss_per_epoch: List[float],
            valid_normal_loss_per_epoch: List[float], 
            valid_anomaly_loss_per_epoch: List[float],
            valid_anomaly_normal_difference_loss_per_epoch: List[float]
        ):
        loss_plot_path = f"{self.distilled_model_folder_path}/loss_plot.png"
        loss_df = pd.DataFrame(
            {
                "epoch": [i for i in range(num_epochs)] + [i for i in range(num_epochs)] + [i for i in range(num_epochs)] + [i for i in range(num_epochs)],
                "loss": train_loss_per_epoch + valid_normal_loss_per_epoch + valid_anomaly_loss_per_epoch + valid_anomaly_normal_difference_loss_per_epoch,
                "split": ["loss_train"] * num_epochs + ["loss_valid_normal"] * num_epochs + ["loss_valid_anomaly"] * num_epochs + ["difference_anomaly_normal"] * num_epochs
            }
        )
        loss_plot = sns.lineplot(loss_df, x="epoch", y="loss", hue="split")
        fig_loss = loss_plot.get_figure()
        fig_loss.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig_loss)

    @staticmethod
    def _save_plot_filters(model, filters_plot_path: str, title: str, model_type: str = "student"):
        if model_type == "student":
            weights = model.embeddings.patch_embeddings.projection.weight
        else:
            # teacher model
            weights = model.patch_embed.proj.weight
        global_max = torch.max(weights).detach().cpu()
        global_min = torch.min(weights).detach().cpu()

        # Let's look at the first 16 filters (out of 384)
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            # Permute from [C, H, W] to [H, W, C] for plotting
            filter_img = weights[i].detach().cpu().permute(1, 2, 0)
            # Normalize weights to [0, 1] for visualization
            normalized_filter = (filter_img - global_min) / (global_max - global_min + 1e-8)
            ax.imshow(normalized_filter)
            ax.axis('off')
            ax.set_title(f"Filter {i}")

        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        fig.savefig(filters_plot_path)
        plt.close(fig)

    def _save_results(self):
        distillation_model_path = os.path.join(self.distilled_model_folder_path, "student_model.pth")
        torch.save(self.student_model.state_dict(), distillation_model_path)

        distillation_optimizer_path = os.path.join(self.distilled_model_folder_path, "optimizer.pth")
        torch.save(self.optimizer.state_dict(), distillation_optimizer_path)

        trained_distillation_config_path = os.path.join(self.distilled_model_folder_path, "config.json")
        shutil.copy2(src=self.train_distillation_config_path, dst=trained_distillation_config_path)

        # Save filters of the trained student model
        filters_plot_path = os.path.join(self.distilled_model_folder_path, "trained_student_filters.png")
        self._save_plot_filters(model=self.student_model, filters_plot_path=filters_plot_path, title="Trained student model")

    def run(self):
        # Save the filters of the student model before training
        untrained_student_filters_plot_path = os.path.join(self.distilled_model_folder_path, "before_training_student_filters.png")
        self._save_plot_filters(model=self.student_model, filters_plot_path=untrained_student_filters_plot_path, title="Before training student model")
        # Save the filters of the teacher model (for reference)
        teacher_filters_plot_path = os.path.join(self.distilled_model_folder_path, "teacher_filters.png")
        self._save_plot_filters(model=self.target_encoder, filters_plot_path=teacher_filters_plot_path, title="Teacher model", model_type="teacher")

        self._load_dataset()
        print(f"Started training for {self.train_distillation_config.num_epochs} epochs...")

        train_loss_per_epoch = []
        valid_anomaly_loss_per_epoch = []
        valid_normal_loss_per_epoch = []
        valid_anomaly_normal_difference_loss_per_epoch = []

        # for early stopping
        best_valid_normal_loss = float("inf")
        best_epoch = -1
        adjusted_num_epochs = self.train_distillation_config.num_epochs

        for epoch in range(self.train_distillation_config.num_epochs):
            print('Epoch %d' % epoch)
            train_losses = []

            self.student_model.train()
            self.target_encoder.eval()
            for train_batch_idx, (image, _, _) in enumerate(self.train_loader):

                image = image.to(self.device)
                with torch.no_grad():
                    # --- Target encoder forward prop
                    target_encoder_emb = self.target_encoder(image)
                    # target_encoder_emb = F.layer_norm(target_encoder_emb, (target_encoder_emb.size(-1),))  # normalize over feature-dim

                # [CLS] token -> position 0; 
                student_emb = self.student_model(image).last_hidden_state[:, 1: , :]
                # student_emb = F.layer_norm(student_emb, (student_emb.size(-1),))  # normalize over feature-dim

                assert target_encoder_emb.shape == student_emb.shape

                self.optimizer.zero_grad()
                loss = self._loss_fn(student=student_emb, teacher=target_encoder_emb)
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    loss_value = loss.detach().cpu().item()
                    train_losses.append(loss_value)
                    if train_batch_idx % 20 == 0:
                        print(f"Train loss batch index {train_batch_idx}: {loss_value}")
        
                    if train_batch_idx % 20 == 0:
                        s_norm = student_emb.detach().norm(dim=-1).mean()
                        t_norm = target_encoder_emb.detach().norm(dim=-1).mean()
                        print(f"Batch {train_batch_idx} -> Student norm: {s_norm:.3f} | Teacher norm: {t_norm:.3f}")

                        cos_sim = F.cosine_similarity(student_emb.detach(), target_encoder_emb.detach(), dim=-1).mean()
                        print(f"Batch {train_batch_idx} -> Cosine similarity: {cos_sim:.3f}")

            avg_train_loss_crt_epoch = np.mean(train_losses)
            train_loss_per_epoch.append(avg_train_loss_crt_epoch)
            print(f"Average train loss epoch {epoch} = {avg_train_loss_crt_epoch}")

            print("Computing the performance on the validation set...")
            self.student_model.eval()
            self.target_encoder.eval()
            with torch.inference_mode():

                valid_losses_anomaly = []
                valid_losses_normal = []

                for valid_batch_idx, (image, label, _, _) in enumerate(self.valid_loader):
                    image = image.to(self.device)

                    with torch.no_grad():
                        # --- Target encoder forward prop
                        target_encoder_emb = self.target_encoder(image)
                        # target_encoder_emb = F.layer_norm(target_encoder_emb, (target_encoder_emb.size(-1),))  # normalize over feature-dim
                        student_emb = self.student_model(image).last_hidden_state[:, 1: , :]
                        # student_emb = F.layer_norm(student_emb, (student_emb.size(-1),))  # normalize over feature-dim

                        assert target_encoder_emb.shape == student_emb.shape

                        valid_loss = self._loss_fn_valid(student=student_emb, teacher=target_encoder_emb).detach().cpu().numpy()
                        label_arr = label.detach().cpu().numpy()

                        anomaly_indices = np.where(label_arr == 1)[0]
                        normal_indices = np.where(label_arr == 0)[0]

                        if len(anomaly_indices) > 0:
                            valid_loss_anomaly = valid_loss[anomaly_indices].mean(axis=(1, 2))
                            valid_losses_anomaly += valid_loss_anomaly.tolist()

                        if len(normal_indices) > 0:
                            valid_loss_normal = valid_loss[normal_indices].mean(axis=(1, 2))
                            valid_losses_normal += valid_loss_normal.tolist()

                assert len(self.valid_dataset) == len(valid_losses_anomaly) + len(valid_losses_normal)

                avg_valid_loss_anomaly_crt_epoch = np.mean(valid_losses_anomaly)
                avg_valid_loss_normal_crt_epoch = np.mean(valid_losses_normal)
                avg_difference_loss_anomaly_normal_crt_epoch = avg_valid_loss_anomaly_crt_epoch - avg_valid_loss_normal_crt_epoch

                print(f"Average validation anomaly loss epoch {epoch} = {avg_valid_loss_anomaly_crt_epoch}")
                print(f"Average validation normal loss epoch {epoch} = {avg_valid_loss_normal_crt_epoch}")
                print(f"Average validation anomaly - normal loss epoch {epoch} = {avg_difference_loss_anomaly_normal_crt_epoch}")
                
                valid_anomaly_loss_per_epoch.append(avg_valid_loss_anomaly_crt_epoch)
                valid_normal_loss_per_epoch.append(avg_valid_loss_normal_crt_epoch)
                valid_anomaly_normal_difference_loss_per_epoch.append(avg_difference_loss_anomaly_normal_crt_epoch)

                # Early stopping
                if avg_valid_loss_normal_crt_epoch < best_valid_normal_loss:
                    best_valid_normal_loss = avg_valid_loss_normal_crt_epoch
                    best_epoch = epoch
                    print(f"Best new checkpoint epoch {epoch} -> saving...")
                    self._save_results()

                else:
                    if (epoch - best_epoch) > self.train_distillation_config.patience:
                        print(f"The validation loss on normal samples did not decrease in the last {self.train_distillation_config.patience}. Stopping the training.")
                        adjusted_num_epochs = epoch + 1
                        break
     
            self.scheduler.step()

        self._save_loss_plot(
            num_epochs=adjusted_num_epochs,
            train_loss_per_epoch=train_loss_per_epoch,
            valid_normal_loss_per_epoch=valid_normal_loss_per_epoch,
            valid_anomaly_loss_per_epoch=valid_anomaly_loss_per_epoch,
            valid_anomaly_normal_difference_loss_per_epoch=valid_anomaly_normal_difference_loss_per_epoch
        )

def main(*args, **kwargs):
    parser = argparse.ArgumentParser(description="Train a student model on the representations of the target encoder")
    parser.add_argument(
        "--train_distillation_config_path", type=str, help="Train distillation configuration file",
    )

    args = parser.parse_args()

    distilled_model_trainer = DistilledModelTrainer(
        train_distillation_config_path=args.train_distillation_config_path,
    )

    distilled_model_trainer.run()


if __name__ == "__main__":
    main()