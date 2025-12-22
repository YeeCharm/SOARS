# ------------------------------------------------------------------------------
# FreeDA
# ------------------------------------------------------------------------------
# Modified from GroupViT (https://github.com/NVlabs/GroupViT)
# Copyright (c) 2021-22, NVIDIA Corporation & affiliates. All Rights Reserved.
# ------------------------------------------------------------------------------
import mmcv
import mmengine
import torch
import numpy as np
# from mmseg.datasets import build_dataloader, build_dataset
from mmseg.registry import DATASETS, TRANSFORMS
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from mmengine.dataset import default_collate, worker_init_fn
from mmengine.dist import get_dist_info
from functools import partial

# from mmseg.datasets.pipelines import Compose
from omegaconf import OmegaConf
from datasets import get_template

# Explicitly import TRANSFORMS to ensure registration
from mmseg.registry import TRANSFORMS

# Define FloatImage class first
class FloatImage:
    def __call__(self, results):
        results['img'] = results['img'].astype(np.float32)
        return results

# Then register it
TRANSFORMS.register_module(module=FloatImage, force=True)

def build_dataset_class_tokens(text_transform, template_set, classnames):
    tokens = []
    templates = get_template(template_set)
    for classname in classnames:
        tokens.append(
            torch.stack([text_transform(template.format(classname)) for template in templates])
        )
    # [N, T, L], N: number of instance, T: number of captions (including ensembled), L: sequence length
    tokens = torch.stack(tokens)

    return tokens


def build_seg_dataset(config):
    """Build a dataset from config."""
    if isinstance(config, str):
        cfg = mmengine.Config.fromfile(config)
    elif hasattr(config, 'config') and config.config:
        # Handle case where config is a DictConfig with a 'config' key pointing to the file
        cfg = mmengine.Config.fromfile(config.config)
    else:
        cfg = config
        
    # mmseg 1.x uses registry
    dataset = DATASETS.build(cfg.data.test)
    return dataset


def build_seg_dataloader(dataset):
    # batch size is set to 1 to handle varying image size (due to different aspect ratio)
    # Re-implement build_dataloader for mmseg 1.x / mmengine
    
    # Use a simple collate function. 
    # Note: mmseg 1.x datasets typically return a dict with 'inputs' and 'data_samples'.
    # We might need a specific collate function if default_collate fails, 
    # but for batch_size=1, default_collate usually works or we can just return the list.
    
    def simple_collate(batch):
        return batch[0] # For batch size 1, just return the item? 
        # Or use default_collate. Let's try default_collate first, but mmseg often uses pseudo_collate for different sized images.
        
    from mmengine.dataset import pseudo_collate
    
    rank, world_size = get_dist_info()
    num_workers = 1
    seed = 42 # Default seed

    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    init_fn = partial(
        worker_init_fn,
        num_workers=num_workers,
        rank=rank,
        seed=seed,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=num_workers,
        shuffle=False if sampler is not None else False,
        collate_fn=pseudo_collate, 
        worker_init_fn=init_fn,
        persistent_workers=True,
        pin_memory=False,
    )
    return data_loader
