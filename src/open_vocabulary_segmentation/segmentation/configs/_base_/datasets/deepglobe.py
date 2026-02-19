# ------------------------------------------------------------------------------
# DeepGlobe Config
# ------------------------------------------------------------------------------
_base_ = ["../custom_import.py"]

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
import mmcv

@DATASETS.register_module(force=True)
class DeepGlobeDataset(BaseSegDataset):
    """DeepGlobe dataset for road extraction."""
    METAINFO = dict(
        classes=('background', 'road'),
        palette=[[0, 0, 0], [255, 255, 255]]
    )

    def __init__(self, img_suffix='.jpg', seg_map_suffix='.png', **kwargs):
        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)

dataset_type = "DeepGlobeDataset"
data_root = "/mnt/sdb/luoyichang/datasets/road_extraction/DeepGlobe_test_1530"

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
        data_prefix=dict(img_path='image_cvt', seg_map_path='label_cvt'),
        pipeline=test_pipeline,
        reduce_zero_label=False
    )
)

test_cfg = dict(mode="slide", stride=(224, 224), crop_size=(448, 448))
