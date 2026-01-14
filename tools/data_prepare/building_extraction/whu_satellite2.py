"""
python datasets/cvt_whu.py data/WHU-BD/val -o data/WHU-BD/val
"""
import argparse
import os
import os.path as osp
import cv2

from mmengine.utils import mkdir_or_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert WHU Satellite-2 dataset to mmsegmentation format')
    parser.add_argument('dataset_path', help='WHU Satellite-2 folder path')
    parser.add_argument('-o', '--out_dir', help='output path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dataset_path = args.dataset_path
    if args.out_dir is None:
        out_dir = osp.join('data', 'WHU-Satellite2')
    else:
        out_dir = args.out_dir

    print('Making directories...')
    mkdir_or_exist(out_dir)
    mkdir_or_exist(osp.join(out_dir, 'image_cvt'))
    mkdir_or_exist(osp.join(out_dir, 'label_cvt'))

    # Process original images: convert .tif to .png
    print('Processing original images...')
    image_path = osp.join(out_dir, 'image')
    for img_name in os.listdir(image_path):
        if img_name.endswith('.tif'):
            # Read image
            img = cv2.imread(osp.join(image_path, img_name))
            # Change extension from .tif to .png
            png_name = img_name.replace('.tif', '.png')
            # Save as PNG
            cv2.imwrite(osp.join(out_dir, 'image_cvt', png_name), img)
            print(f'Converted {img_name} to {png_name}')

    # Process label images: binarize
    print('Processing label images...')
    label_path = osp.join(out_dir, 'label')
    for img_name in os.listdir(label_path):
        if img_name.endswith('.tif'):
            # Read as grayscale
            img = cv2.imread(osp.join(label_path, img_name), cv2.IMREAD_GRAYSCALE)
            # Binarize
            img[img < 128] = 0
            img[img >= 128] = 1
            # Change extension from .tif to .png
            png_name = img_name.replace('.tif', '.png')
            # Save as PNG
            cv2.imwrite(osp.join(out_dir, 'label_cvt', png_name), img)
            print(f'Binarized {img_name} to {png_name}')

    print('Done!')


if __name__ == '__main__':
    main()
