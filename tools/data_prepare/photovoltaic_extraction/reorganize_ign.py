"""
python tools/reorganize_ign.py ../ign

This script processes bdappv_ign dataset by:
1. Reading mask files from ign/mask, normalizing them, and saving to ign/processed/mask_cvt
2. Copying corresponding image files from ign/img to ign/processed/img_cvt
"""
import argparse
import os
import os.path as osp
import cv2

from mmengine.utils import mkdir_or_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process bdappv_ign dataset')
    parser.add_argument('dataset_path', help='ign folder path (relative or absolute)')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dataset_path = args.dataset_path
    
    out_dir = osp.join(dataset_path, 'processed')
    mask_cvt_dir = osp.join(out_dir, 'mask_cvt')
    img_cvt_dir = osp.join(out_dir, 'img_cvt')
    
    mask_path = osp.join(dataset_path, 'mask')
    img_path = osp.join(dataset_path, 'img')

    print('Making directories...')
    mkdir_or_exist(mask_cvt_dir)
    mkdir_or_exist(img_cvt_dir)

    print(f'Processing masks from {mask_path}...')
    for mask_name in os.listdir(mask_path):
        mask = cv2.imread(osp.join(mask_path, mask_name), cv2.IMREAD_GRAYSCALE)
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        cv2.imwrite(osp.join(mask_cvt_dir, mask_name), mask)
        
        img_file = osp.join(img_path, mask_name)
        if osp.exists(img_file):
            img = cv2.imread(img_file)
            cv2.imwrite(osp.join(img_cvt_dir, mask_name), img)
        else:
            print(f'Warning: Image {mask_name} not found in {img_path}')

    print('Done!')


if __name__ == '__main__':
    main()
