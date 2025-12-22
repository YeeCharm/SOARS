# ------------------------------------------------------------------------------
# OpenEarthMap Config (Final Fix for Registry & Float Type)
# ------------------------------------------------------------------------------
_base_ = ["../custom_import.py"]

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
import mmcv

# Force=True prevents registry conflict errors
@DATASETS.register_module(force=True)
class OpenEarthMapDataset(BaseSegDataset):
    """OpenEarthMap dataset."""
    METAINFO = dict(
        classes=(
            'background', 'bareland', 'grass', 'pavement', 'road', 
            'tree', 'water', 'cropland', 'building'
        ),
        palette=[
            [0, 0, 0], [128, 64, 128], [0, 255, 0], [60, 40, 222],
            [128, 128, 128], [0, 128, 0], [0, 0, 255], [255, 255, 0], [255, 0, 0]
        ]
    )

    def __init__(self, img_suffix='.tif', seg_map_suffix='.tif', **kwargs):
        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)

dataset_type = "OpenEarthMapDataset"
data_root = "/mnt/sdb/luoyichang/datasets/OpenEarthMap"

# --- Fix: Use standard ImageToTensor or PackSegInputs logic ---
# We remove 'CastTo' as it is causing registry errors.
# 'PackSegInputs' handles formatting, but we need to ensure float32 before that.
# Using a simple Normalize with identity (or standard stats) is the most robust way in mmseg to force float.

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    # dict(type="Resize", scale=(448, 448), keep_ratio=True),
    dict(type="Resize", scale=(1024, 1024), keep_ratio=True),
    # Adding Normalize is the standard way to convert uint8 -> float32 in MMSegmentation
    dict(type='Normalize', **img_norm_cfg),
    dict(type="PackSegInputs")
]

data = dict(
    test=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=test_pipeline,
        reduce_zero_label=False
    )
)

# test_cfg = dict(mode="slide", stride=(224, 224), crop_size=(448, 448))
test_cfg = dict(mode="slide", stride=(300, 300), crop_size=(448, 448))