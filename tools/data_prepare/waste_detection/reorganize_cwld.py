"""
python tools/reorganize_cwld.py ../Original_Dataset

This script processes CWLD dataset by:
1. Reading image files from Changping_District_Beijing/images and Daxing_District_Beijing/images
   Converting from tif to png format and saving to processed/img_cvt
2. Processing label files from Changping_District_Beijing/label and Daxing_District_Beijing/label
   Extracting Red (255,0,0) regions as class 1, setting everything else to class 0.
   Saving as single-channel PNG (mask) to processed/label_cvt
"""
import argparse
import os
import os.path as osp
import cv2
import numpy as np

from mmengine.utils import mkdir_or_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process CWLD dataset')
    parser.add_argument('dataset_path', help='Original_Dataset folder path (relative or absolute)')
    args = parser.parse_args()
    return args


def process_label(label):
    """
    Input: label image in BGR format (H, W, 3)
    Output: mask image (H, W) with values 0 and 1
    Logic: 
      - Target Class (Waste): Red pixels RGB(255, 0, 0) -> BGR(0, 0, 255)
      - Background: All other pixels
    """
    # 创建一个全黑的单通道掩码，默认值为 0 (Background)
    mask = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
    
    # 提取红色像素：在 OpenCV BGR 格式中，Red是通道2
    # 垃圾倾倒区颜色为 RGB(255, 0, 0)  => BGR(0, 0, 255)
    is_red = (label[:, :, 0] == 0) & (label[:, :, 1] == 0) & (label[:, :, 2] == 255)
    
    # 将红色区域标记为 1 (Target)
    mask[is_red] = 1
    
    return mask


def main():
    args = parse_args()
    dataset_path = args.dataset_path
    
    out_dir = osp.join(osp.dirname(dataset_path), 'processed')
    img_cvt_dir = osp.join(out_dir, 'img_cvt')
    label_cvt_dir = osp.join(out_dir, 'label_cvt')
    
    img_dirs = [
        osp.join(dataset_path, 'Changping_District_Beijing', 'images'),
        osp.join(dataset_path, 'Daxing_District_Beijing', 'images')
    ]
    label_dirs = [
        osp.join(dataset_path, 'Changping_District_Beijing', 'label'),
        osp.join(dataset_path, 'Daxing_District_Beijing', 'label')
    ]

    print('Making directories...')
    mkdir_or_exist(img_cvt_dir)
    mkdir_or_exist(label_cvt_dir)

    print('Processing images...')
    for img_dir in img_dirs:
        if not osp.exists(img_dir):
            print(f'Warning: {img_dir} does not exist, skipping...')
            continue
        print(f'Processing images from {img_dir}...')
        for img_name in os.listdir(img_dir):
            if img_name.lower().endswith('.tif'):
                img = cv2.imread(osp.join(img_dir, img_name))
                if img is not None:
                    new_name = osp.splitext(img_name)[0] + '.png'
                    cv2.imwrite(osp.join(img_cvt_dir, new_name), img)
                else:
                    print(f'Warning: Failed to read {img_name}')

    print('Processing labels...')
    for label_dir in label_dirs:
        if not osp.exists(label_dir):
            print(f'Warning: {label_dir} does not exist, skipping...')
            continue
        print(f'Processing labels from {label_dir}...')
        for label_name in os.listdir(label_dir):
            # cv2.imread 默认读取为 BGR 格式
            label = cv2.imread(osp.join(label_dir, label_name))
            if label is not None:
                # 处理标签：红->1，其他->0，返回单通道 Mask
                processed_mask = process_label(label)
                new_name = osp.splitext(label_name)[0] + '.png'
                # 保存为单通道 PNG
                cv2.imwrite(osp.join(label_cvt_dir, new_name), processed_mask)
            else:
                print(f'Warning: Failed to read {label_name}')

    print('Done!')


if __name__ == '__main__':
    main()
