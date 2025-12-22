# ------------------------------------------------------------------------------
# FreeDA
# ------------------------------------------------------------------------------
# Modified from GroupViT (https://github.com/NVlabs/GroupViT)
# Copyright (c) 2021-22, NVIDIA Corporation & affiliates. All Rights Reserved.
# ------------------------------------------------------------------------------
import mmcv
import mmengine
import torch
# from mmseg.datasets import build_dataloader, build_dataset
# from mmseg.datasets.pipelines import Compose
from omegaconf import OmegaConf
from datasets import get_template

from .dinotext_seg import DINOTextSegInference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_dinotext_seg_inference(
    model,
    dataset,
    config,
    seg_config,
):
    if isinstance(seg_config, str):
        dset_cfg = mmengine.Config.fromfile(seg_config)  # dataset config
    else:
        dset_cfg = seg_config
        
    if hasattr(dataset.dataset, 'CLASSES'):
        classes = dataset.dataset.CLASSES
    else:
        classes = dataset.dataset.METAINFO['classes']

    with_bg = classes[0] == "background"
    if with_bg:
        classnames = classes[1:]
    else:
        classnames = classes
    text_tokens = model.build_dataset_class_tokens(config.evaluate.template, classnames)
    text_embedding = model.build_text_embedding(text_tokens)
    kwargs = dict(with_bg=with_bg)
    if hasattr(dset_cfg, "test_cfg"):
        kwargs["test_cfg"] = dset_cfg.test_cfg

    model_type = config.model.type
    if model_type == "DINOText":
        seg_model = DINOTextSegInference(model, text_embedding, classnames, **kwargs, **config.evaluate)
    else:
        raise ValueError(model_type)

    seg_model.CLASSES = classes
    if hasattr(dataset.dataset, 'PALETTE'):
        seg_model.PALETTE = dataset.dataset.PALETTE
    else:
        seg_model.PALETTE = dataset.dataset.METAINFO['palette']

    return seg_model
