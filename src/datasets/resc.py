# Code adapted from: https://github.com/facebookresearch/ijepa

import os
import glob
from PIL import Image
from logging import getLogger

import torch
import torchvision
from torch.utils.data import Dataset

logger = getLogger()


def make_resc(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    drop_last=False,
    is_train=True
):
    if is_train:
        dataset = RESC(
            root=root_path,
            image_folder=image_folder,
            transform=transform)
        logger.info('RESC dataset created')
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset=dataset,
            num_replicas=world_size,
            rank=rank)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=collator,
            sampler=dist_sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            pin_memory=pin_mem,
            num_workers=num_workers,
            persistent_workers=False,
        )
    else:
        dataset = ImageDataset(
            root=root_path,
            image_folder=image_folder,
            transform=transform
        )
        logger.info('RESC dataset created')
        dist_sampler = None
        data_loader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=collator,
            batch_size=batch_size,
            drop_last=drop_last,
            pin_memory=pin_mem,
            num_workers=num_workers,
            persistent_workers=False,
            shuffle=False
        )

    logger.info('RESC unsupervised data loader created')

    return dataset, data_loader, dist_sampler


class RESC(torchvision.datasets.ImageFolder):
    def __init__(
        self,
        root,
        image_folder='RESC/train/',
        transform=None
    ):
        """
        RESC

        :param root: root directory for RESC data
        :param image_folder: path to images inside root directory
        """

        data_path = os.path.join(root, image_folder)
        logger.info(f'data-path {data_path}')

        super(RESC, self).__init__(root=data_path, transform=transform)
        logger.info('Initialized RESC dataset')


class ImageDataset(Dataset):
    def __init__(self, root, image_folder, transform=None):
        self.root = root
        self.image_folder = image_folder
        self.transform = transform

        # images
        self.image_paths = sorted(glob.glob(os.path.join(self.root, self.image_folder, "*.png")))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        # apply transformations
        if self.transform:
            image = self.transform(image)

        return image