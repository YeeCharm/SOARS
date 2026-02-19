# ------------------------------------------------------------------------------
# WBS Config
# ------------------------------------------------------------------------------
_base_ = ["../custom_import.py"]

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
import mmcv

@DATASETS.register_module(force=True)
class WBSDataset(BaseSegDataset):
    """WBS dataset for water body segmentation."""
    METAINFO = dict(
        classes=('background', 'water'),
        palette=[[0, 0, 0], [255, 255, 255]]
    )

    def __init__(self, img_suffix='.jpg', seg_map_suffix='.jpg', **kwargs):
        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)

dataset_type = "WBSDataset"
data_root = "/mnt/sdb/luoyichang/datasets/water_extraction/processed"

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", scale=(448, 448), keep_ratio=True),
    dict(type='FloatImage'),
    dict(type="PackSegInputs")
]

data = dict(
    test=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images_cvt', seg_map_path='masks_cvt'),
        pipeline=test_pipeline,
        reduce_zero_label=False
    )
)

test_cfg = dict(mode="slide", stride=(224, 224), crop_size=(448, 448))
