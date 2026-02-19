# ------------------------------------------------------------------------------
# OpenEarthMap Config (No Background Class)
# ------------------------------------------------------------------------------
_base_ = ["../custom_import.py"]

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
import mmcv

@DATASETS.register_module(force=True)
class OpenEarthMapDataset(BaseSegDataset):
    """OpenEarthMap dataset (without background class).
    
    In segmentation map annotation for OpenEarthMap, 0 is background.
    We use reduce_zero_label=True to map labels 1-8 to 0-7.
    """
    METAINFO = dict(
        classes=('bareland,wasteland,bare soil', 'rangeland,pastureland,pasture', 'developed space,impervious surface', 'road,pavement', 
                 'tree', 'water', 'farmland', 'building'),
        palette=[[128, 0, 0], [0, 255, 36], [148, 148, 148], 
                 [255, 255, 255], [34, 97, 38], [0, 69, 255], [75, 181, 73], 
                 [222, 31, 7]],
       
    )

    def __init__(self, img_suffix='.tif', seg_map_suffix='.tif', reduce_zero_label=False, **kwargs):
        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, reduce_zero_label=reduce_zero_label, **kwargs)

dataset_type = "OpenEarthMapDataset"
data_root = "/mnt/sdb/luoyichang/datasets/OpenEarthMap"

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
