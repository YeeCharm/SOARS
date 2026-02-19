# ------------------------------------------------------------------------------
# FreeDA
# ------------------------------------------------------------------------------
# Modified from GroupViT (https://github.com/NVlabs/GroupViT)
# Copyright (c) 2021-22, NVIDIA Corporation & affiliates. All Rights Reserved.
# ------------------------------------------------------------------------------
_base_ = ["../custom_import.py"]
# dataset settings
dataset_type = "LoveDADataset"
data_root = "/mnt/sdb/luoyichang/datasets/data/loveDA"
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
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=test_pipeline,
    )
)

# test_cfg = dict(mode="whole")
test_cfg = dict(mode="slide", stride=(224, 224), crop_size=(448, 448))
