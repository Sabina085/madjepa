# Code adapted from: https://github.com/facebookresearch/ijepa

import argparse
import os
import logging

import multiprocessing as mp
from src.utils.distributed import init_distributed
from src.train import main as app_main

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='path to the config file to load',
    default='configs.yaml')
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='which devices to use on local machine')


def process_main(rank, fname, world_size, devices):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])
    logging.basicConfig()

    logger = logging.getLogger()
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f'called-params {fname}')

    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f'Running... (rank: {rank}/{world_size})')
    app_main(config_path=fname)


if __name__ == '__main__':
    args = parser.parse_args()

    num_gpus = len(args.devices)
    mp.set_start_method('spawn')

    for rank in range(num_gpus):
        mp.Process(
            target=process_main,
            args=(rank, args.fname, num_gpus, args.devices)
        ).start()
