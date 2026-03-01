# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from logging import getLogger

import torch
import torchvision

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
    drop_last=True
):
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

        :param root: root network directory for RESC data
        :param image_folder: path to images inside root network directory
        """

        data_path = os.path.join(root, image_folder)
        logger.info(f'data-path {data_path}')

        super(RESC, self).__init__(root=data_path, transform=transform)
        logger.info('Initialized RESC dataset')

