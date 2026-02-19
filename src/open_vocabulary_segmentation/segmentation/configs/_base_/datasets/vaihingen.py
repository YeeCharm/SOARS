# ------------------------------------------------------------------------------
# ISPRS Vaihingen Config
# ------------------------------------------------------------------------------
_base_ = ["../custom_import.py"]

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
import mmcv

@DATASETS.register_module(force=True)
class ISPRSDataset(BaseSegDataset):
    """ISPRS dataset for remote sensing semantic segmentation.
    
    In segmentation map annotation for ISPRS, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True.
    """
    METAINFO = dict(
        classes=('road, parking lot', 'building', 'low vegetation', 
                            'tree', 'car', 'clutter, background'),
        palette=[[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], 
                 [255, 255, 0], [255, 0, 0]],
    )

    def __init__(self, img_suffix='.png', seg_map_suffix='.png', reduce_zero_label=True, **kwargs):
        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, reduce_zero_label=reduce_zero_label, **kwargs)

dataset_type = "ISPRSDataset"
data_root = "/mnt/sdb/luoyichang/datasets/ISPRS_Vaihingen/vaihingen"

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
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=test_pipeline,
        reduce_zero_label=True
    )
)

test_cfg = dict(mode="slide", stride=(224, 224), crop_size=(448, 448))
