import os
import os.path as osp
import numpy as np
from PIL import Image
from mmengine.utils import mkdir_or_exist

# ==========================================
# 改动1: 标签 ID 从 1-based (1-17) 改为 0-based (0-16)
# 原因: 深度学习模型(如mmseg)要求类别从0开始。
# 如果用1-17，模型会认为有18类(0-17)，导致维度不匹配或NaN。
# ==========================================
COLOR_MAP = {
    0: [166, 202, 240],   # airplane (原Label 1 -> 0)
    1: [128, 128, 0],     # bare soil
    2: [0, 0, 128],       # buildings
    3: [255, 0, 0],       # cars
    4: [0, 128, 0],       # chaparral
    5: [128, 0, 0],       # court
    6: [255, 233, 233],   # dock
    7: [160, 160, 164],   # field
    8: [0, 128, 128],     # grass
    9: [90, 87, 255],     # mobile home
    10: [255, 255, 0],    # pavement
    11: [255, 192, 0],    # sand
    12: [0, 0, 255],      # sea
    13: [255, 0, 192],    # ship
    14: [128, 0, 128],    # tanks
    15: [0, 255, 0],      # trees
    16: [0, 255, 255],    # water (原Label 17 -> 16)
}

def convert_tif_to_png():
    """
    将原始 Images 文件夹下的 .tif 转换为 .png
    """
    base_dir = osp.dirname(osp.abspath(__file__))
    base_dir = osp.dirname(base_dir)
    
    images_dir = osp.join(base_dir, 'Images')
    output_dir = osp.join(base_dir, 'processed', 'images')
    
    mkdir_or_exist(output_dir)
    
    # 检查原始目录是否存在
    if not osp.exists(images_dir):
        print(f"Error: {images_dir} not found!")
        return

    subfolders = [f for f in os.listdir(images_dir) 
                  if osp.isdir(osp.join(images_dir, f))]
    
    print(f'Found {len(subfolders)} subfolders in Images directory')
    
    for subfolder in subfolders:
        src_path = osp.join(images_dir, subfolder)
        tif_files = [f for f in os.listdir(src_path) if f.endswith('.tif')]
        
        # 保持输出目录结构 (可选，如果不需要子文件夹可以去掉 osp.join(output_dir, subfolder))
        # 这里为了简单，还是全部放在 output_dir 或者按子文件夹放
        # 为了适配多数分割数据集格式，通常建议打平或者保持同构。
        # 你的原代码是打平放到 output_dir，这里保持一致。
        
        print(f'Processing {subfolder}: {len(tif_files)} tif files')
        
        for tif_file in tif_files:
            src_file = osp.join(src_path, tif_file)
            # 文件名处理：建议加上文件夹前缀防止重名，例如 agricultural_01.png
            # DLRSD原名通常已包含类别，如 "agricultural01.tif"
            png_file = osp.splitext(tif_file)[0] + '.png'
            dst_file = osp.join(output_dir, png_file)
            
            img = Image.open(src_file)
            img = img.convert('RGB')
            img.save(dst_file, 'PNG')
    
    print(f'All images converted to {output_dir}')


def rgb_to_single_channel(rgb_img):
    rgb_array = np.array(rgb_img)
    h, w, _ = rgb_array.shape
    
    # ==========================================
    # 改动2: 初始化由 0 改为 255 (Ignore Index)
    # 原因: 原代码初始化为0。如果某个像素颜色没匹配上(比如边缘杂色)，
    # 它会被标记为0。而在改动1后，0代表"飞机"。
    # 这会导致大量杂色被错误训练为飞机。
    # 255是通用的"忽略标签"，计算Loss时会跳过这些像素。
    # ==========================================
    single_channel = np.full((h, w), 255, dtype=np.uint8)
    
    for class_id, color in COLOR_MAP.items():
        color_array = np.array(color)
        # 匹配颜色
        mask = np.all(rgb_array == color_array, axis=-1)
        single_channel[mask] = class_id
    
    return single_channel


def convert_labels():
    """
    将原始 Labels (RGB) 转换为 单通道索引图 (TrainID 0-16)
    """
    base_dir = osp.dirname(osp.abspath(__file__))
    base_dir = osp.dirname(base_dir)
    
    labels_dir = osp.join(base_dir, 'Labels')
    output_dir = osp.join(base_dir, 'processed', 'labels')
    
    mkdir_or_exist(output_dir)
    
    if not osp.exists(labels_dir):
        print(f"Error: {labels_dir} not found!")
        return

    subfolders = [f for f in os.listdir(labels_dir) 
                  if osp.isdir(osp.join(labels_dir, f))]
    
    print(f'Found {len(subfolders)} subfolders in Labels directory')
    
    total_files = 0
    for subfolder in subfolders:
        src_path = osp.join(labels_dir, subfolder)
        
        # ==========================================
        # 改动3: 增加对 .tif 后缀的支持
        # 原因: DLRSD原始Label通常是tif格式。你的原代码只读png。
        # 如果你没先手动转过格式，原代码会读不到文件。
        # ==========================================
        label_files = [f for f in os.listdir(src_path) if f.endswith('.png') or f.endswith('.tif')]
        
        print(f'Processing {subfolder}: {len(label_files)} label files')
        
        for label_file in label_files:
            src_file = osp.join(src_path, label_file)
            dst_file = osp.join(output_dir, osp.splitext(label_file)[0] + '.png')
            
            # 打开图片并强制转为RGB (防止某些图是RGBA或P模式)
            rgb_img = Image.open(src_file).convert('RGB')
            
            # 核心转换逻辑
            single_channel = rgb_to_single_channel(rgb_img)
            
            # 保存为单通道PNG (模式'L')
            # 这一步很重要，确保保存的是灰度图而不是RGB图
            result_img = Image.fromarray(single_channel, mode='L')
            result_img.save(dst_file, 'PNG')
            
            total_files += 1
    
    print(f'All labels converted to {output_dir}')
    print(f'Total label files: {total_files}')


if __name__ == '__main__':
    print('=== Converting Images (TIF -> PNG) ===')
    convert_tif_to_png()
    print()
    print('=== Converting Labels (RGB -> Single Channel ID) ===')
    convert_labels()
    print()
    print('All conversions completed!')
