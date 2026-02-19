# ------------------------------------------------------------------------------
# WHU Aerial Config
# ------------------------------------------------------------------------------
_base_ = ["../custom_import.py"]

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
import mmcv

@DATASETS.register_module(force=True)
class WHUAerialDataset(BaseSegDataset):
    """WHU Aerial dataset for building extraction."""
    METAINFO = dict(
        classes=('background', 'building'),
        palette=[[0, 0, 0], [255, 255, 255]]
    )

    def __init__(self, img_suffix='.png', seg_map_suffix='.png', **kwargs):
        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)

dataset_type = "WHUAerialDataset"
data_root = "/mnt/sdb/luoyichang/datasets/building_extraction/WHU_Building_Dataset/Aerial/processed"

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
        data_prefix=dict(img_path='images', seg_map_path='label_cvt'),
        pipeline=test_pipeline,
        reduce_zero_label=False
    )
)

test_cfg = dict(mode="slide", stride=(224, 224), crop_size=(448, 448))
