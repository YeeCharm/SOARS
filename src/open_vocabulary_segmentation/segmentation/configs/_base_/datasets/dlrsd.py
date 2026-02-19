# ------------------------------------------------------------------------------
# DLRSD Config
# ------------------------------------------------------------------------------
_base_ = ["../custom_import.py"]

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
import mmcv

@DATASETS.register_module(force=True)
class DLRSDataset(BaseSegDataset):
    """DLRSD dataset for remote sensing scene classification."""
    METAINFO = dict(
        classes=('airplane', 'bare soil', 'buildings', 'cars', 'chaparral', 
                 'court', 'dock', 'field', 'grass', 'mobile home', 
                 'pavement', 'sand', 'sea', 'ship', 'tanks', 'trees', 'water'),
        palette=[[166, 202, 240], [128, 128, 0], [0, 0, 128], [255, 0, 0], 
                 [0, 128, 0], [128, 0, 0], [255, 233, 233], [160, 160, 164], 
                 [0, 128, 128], [90, 87, 255], [255, 255, 0], [255, 192, 0], 
                 [0, 0, 255], [255, 0, 192], [128, 0, 128], [0, 255, 0], 
                 [0, 255, 255]]
    )

    def __init__(self, img_suffix='.png', seg_map_suffix='.png', **kwargs):
        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)

dataset_type = "DLRSDataset"
data_root = "/mnt/sdb/luoyichang/datasets/DLRSD"

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
        data_prefix=dict(img_path='processed/images', seg_map_path='processed/labels'),
        pipeline=test_pipeline,
        reduce_zero_label=False
    )
)

test_cfg = dict(mode="slide", stride=(224, 224), crop_size=(448, 448))
