# ------------------------------------------------------------------------------
# BDAPPV-IGN Config
# ------------------------------------------------------------------------------
_base_ = ["../custom_import.py"]

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
import mmcv

@DATASETS.register_module(force=True)
class BDAPPVIGNDataset(BaseSegDataset):
    """BDAPPV-IGN dataset for solar photovoltaic panel segmentation."""
    METAINFO = dict(
        classes=('background', 'solar_panel'),
        palette=[[0, 0, 0], [255, 255, 255]]
    )

    def __init__(self, img_suffix='.png', seg_map_suffix='.png', **kwargs):
        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)

dataset_type = "BDAPPVIGNDataset"
data_root = "/mnt/sdb/luoyichang/datasets/solar_photovoltaic/bdappv/ign/processed"

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
        data_prefix=dict(img_path='img_cvt', seg_map_path='mask_cvt'),
        pipeline=test_pipeline,
        reduce_zero_label=False
    )
)

test_cfg = dict(mode="slide", stride=(224, 224), crop_size=(448, 448))
