# ------------------------------------------------------------------------------
# WHDLD Config
# ------------------------------------------------------------------------------
_base_ = ["../custom_import.py"]

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
import mmcv

@DATASETS.register_module(force=True)
class WHDLDDataset(BaseSegDataset):
    """WHDLD dataset for remote sensing land cover classification."""
    METAINFO = dict(
        classes=('building', 'road', 'pavement', 'vegetation', 'bare_soil', 'water'),
        palette=[[255, 0, 0], [255, 255, 0], [192, 192, 0], [0, 255, 0], 
                 [128, 128, 128], [0, 0, 255]]
    )

    def __init__(self, img_suffix='.jpg', seg_map_suffix='.png', **kwargs):
        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)

dataset_type = "WHDLDDataset"
data_root = "/mnt/sdb/luoyichang/datasets/WHDLD/processed"

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
        data_prefix=dict(img_path='images', seg_map_path='labels'),
        pipeline=test_pipeline,
        reduce_zero_label=False
    )
)

test_cfg = dict(mode="slide", stride=(224, 224), crop_size=(448, 448))
