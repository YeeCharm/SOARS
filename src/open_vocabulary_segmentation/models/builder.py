# ------------------------------------------------------------------------------
# FreeDA
# ------------------------------------------------------------------------------
# Modified from GroupViT (https://github.com/NVlabs/GroupViT)
# Copyright (c) 2021-22, NVIDIA Corporation & affiliates. All Rights Reserved.
# ------------------------------------------------------------------------------
# from mmcv.utils import Registry
from omegaconf import OmegaConf

try:
    # 新版本 OpenMMLab（mmcv 2.x + mmengine）
    from mmengine.registry import Registry
except ImportError:
    # 老版本兼容（mmcv 1.x）
    from mmcv.utils import Registry


MODELS = Registry("model")


def build_model(config):
    model = MODELS.build(OmegaConf.to_container(config, resolve=True))
    return model
