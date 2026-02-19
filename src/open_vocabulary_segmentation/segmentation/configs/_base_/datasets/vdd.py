# ------------------------------------------------------------------------------
# FreeDA
# ------------------------------------------------------------------------------
# Modified from GroupViT (https://github.com/NVlabs/GroupViT)
# Copyright (c) 2021-22, NVIDIA Corporation & affiliates.
# ------------------------------------------------------------------------------

_base_ = ["../custom_import.py"]

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
import mmcv

# Force=True prevents registry conflict errors
@DATASETS.register_module(force=True)
class VDDataset(BaseSegDataset):
    """VDD dataset."""
    METAINFO = dict(
        classes=(
            'other', 'building facade', 'road', 'vegetation', 'vehicle', 'roof', 'water'
        ),
        palette=[
            [0, 0, 0], [102, 102, 156], [128, 64, 128], [107, 142, 35],
               [0, 0, 142], [70, 70, 70], [0, 200, 200]
        ]
    )
    

    def __init__(self, img_suffix='.JPG', seg_map_suffix='.png', **kwargs):
        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)

dataset_type = "VDDataset"
data_root = "/mnt/sdb/luoyichang/datasets/VDD"

# img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", scale=(448, 448), keep_ratio=True),
    dict(type="FloatImage"),
    dict(type="PackSegInputs")
]

data = dict(
    test=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='test/src', seg_map_path='test/gt'),
        pipeline=test_pipeline,
        reduce_zero_label=False
    )
)

# test_cfg = dict(mode="whole")
test_cfg = dict(mode="slide", stride=(224, 224), crop_size=(448, 448))
