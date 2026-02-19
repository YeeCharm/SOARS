import os
import os.path as osp
import shutil
import numpy as np
from PIL import Image
from mmengine.utils import mkdir_or_exist

COLOR_MAP = {
    0: [255, 0, 0],
    1: [255, 255, 0],
    2: [192, 192, 0],
    3: [0, 255, 0],
    4: [128, 128, 128],
    5: [0, 0, 255],
}

def copy_images():
    base_dir = osp.dirname(osp.abspath(__file__))
    base_dir = osp.dirname(base_dir)
    
    images_dir = osp.join(base_dir, 'Images')
    output_dir = osp.join(base_dir, 'processed', 'images')
    
    mkdir_or_exist(output_dir)
    
    if not osp.exists(images_dir):
        print(f"Error: {images_dir} not found!")
        return

    image_files = [f for f in os.listdir(images_dir) if osp.isfile(osp.join(images_dir, f))]
    
    print(f'Found {len(image_files)} image files in Images directory')
    
    for image_file in image_files:
        src_file = osp.join(images_dir, image_file)
        dst_file = osp.join(output_dir, image_file)
        shutil.copy2(src_file, dst_file)
    
    print(f'All images copied to {output_dir}')
    print(f'Total image files: {len(image_files)}')


def rgb_to_single_channel(rgb_img):
    rgb_array = np.array(rgb_img)
    h, w, _ = rgb_array.shape
    
    single_channel = np.full((h, w), 255, dtype=np.uint8)
    
    for class_id, color in COLOR_MAP.items():
        color_array = np.array(color)
        mask = np.all(rgb_array == color_array, axis=-1)
        single_channel[mask] = class_id
    
    return single_channel


def convert_labels():
    base_dir = osp.dirname(osp.abspath(__file__))
    base_dir = osp.dirname(base_dir)
    
    labels_dir = osp.join(base_dir, 'Labels')
    output_dir = osp.join(base_dir, 'processed', 'labels')
    
    mkdir_or_exist(output_dir)
    
    if not osp.exists(labels_dir):
        print(f"Error: {labels_dir} not found!")
        return

    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.png') or f.endswith('.tif')]
    
    print(f'Found {len(label_files)} label files in Labels directory')
    
    for label_file in label_files:
        src_file = osp.join(labels_dir, label_file)
        dst_file = osp.join(output_dir, osp.splitext(label_file)[0] + '.png')
        
        rgb_img = Image.open(src_file).convert('RGB')
        
        single_channel = rgb_to_single_channel(rgb_img)
        
        result_img = Image.fromarray(single_channel, mode='L')
        result_img.save(dst_file, 'PNG')
    
    print(f'All labels converted to {output_dir}')
    print(f'Total label files: {len(label_files)}')


if __name__ == '__main__':
    print('=== Copying Images ===')
    copy_images()
    print()
    print('=== Converting Labels (RGB -> Single Channel ID) ===')
    convert_labels()
    print()
    print('All conversions completed!')
