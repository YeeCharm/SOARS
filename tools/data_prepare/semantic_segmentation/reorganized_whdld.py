import argparse
import os
import os.path as osp
import shutil
from pathlib import Path
from tqdm import tqdm  # 如果没有tqdm，可以删掉这行和下面的tqdm包装

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert OpenEarthMap dataset to mmsegmentation format')
    
    # 你的原始数据路径
    parser.add_argument('--dataset_path', 
                        default='/mnt/sdb/luoyichang/datasets/OpenEarthMap/OpenEarthMap_wo_xBD', 
                        help='Original OpenEarthMap folder path')
    
    # 你想要输出的目标路径（上一级目录）
    parser.add_argument('--out_dir', 
                        default='/mnt/sdb/luoyichang/datasets/OpenEarthMap', 
                        help='Output path for reorganized data')
    
    # 指定划分文件名为 val.txt
    parser.add_argument('--split_file', 
                        default='val.txt', 
                        help='Name of the validation split file')
    
    args = parser.parse_args()
    return args

def mkdir_or_exist(dir_name):
    os.makedirs(dir_name, exist_ok=True)

def main():
    args = parse_args()
    dataset_path = args.dataset_path
    out_dir = args.out_dir
    split_file_path = osp.join(dataset_path, args.split_file)

    print(f'Source Path: {dataset_path}')
    print(f'Target Path: {out_dir}')
    print(f'Split File:  {split_file_path}')

    if not osp.exists(split_file_path):
        raise FileNotFoundError(f"找不到划分文件: {split_file_path}，请确认文件名是否正确。")

    # 1. 创建目标文件夹结构
    print('Making directories...')
    mkdir_or_exist(out_dir)
    mkdir_or_exist(osp.join(out_dir, 'img_dir', 'val'))
    mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'val'))

    # 2. 读取 val.txt 中的文件名列表
    print('Reading split file...')
    with open(split_file_path, 'r') as f:
        # 读取每一行，去掉换行符，组成一个集合以便快速查找
        # 文件内容示例: aachen_11.tif
        val_filenames = set(line.strip() for line in f.readlines() if line.strip())
    
    print(f'Total validation images to process: {len(val_filenames)}')

    # 3. 扫描原始目录下的所有图片
    print('Scanning all images in source directory (this may take a moment)...')
    # 递归查找所有 .tif 文件，并且路径中包含 /images/ 的才算作原图
    # 假设结构是: region_name/images/xxx.tif
    all_image_paths = [p for p in Path(dataset_path).rglob("*.tif") if "/images/" in str(p)]

    print(f'Found {len(all_image_paths)} total images in source. Filtering for validation set...')

    processed_count = 0
    missing_labels = 0

    # 4. 遍历并复制
    # 使用 tqdm 显示进度条，如果报错 no module named tqdm，把 tqdm(all_image_paths) 改为 all_image_paths 即可
    for img_path_obj in tqdm(all_image_paths, desc="Copying files"):
        file_name = img_path_obj.name # 例如 aachen_11.tif
        
        # 核心逻辑：只有在 val.txt 名单里的图片才处理
        if file_name in val_filenames:
            src_img_path = str(img_path_obj)
            
            # 推导标签路径：把路径中的 /images/ 替换为 /labels/
            # OpenEarthMap结构通常是 .../images/foo.tif 对应 .../labels/foo.tif
            src_ann_path = src_img_path.replace('/images/', '/labels/')
            
            # 目标路径
            dst_img_path = osp.join(out_dir, 'img_dir', 'val', file_name)
            dst_ann_path = osp.join(out_dir, 'ann_dir', 'val', file_name)
            
            # 执行复制
            shutil.copy(src_img_path, dst_img_path)
            
            if os.path.exists(src_ann_path):
                shutil.copy(src_ann_path, dst_ann_path)
            else:
                # 这种情况很少见，但为了防止报错，记录一下
                print(f"\nWarning: Label not found for {file_name}")
                missing_labels += 1
            
            processed_count += 1
            
            # 从集合中移除已处理的，方便最后检查是否有遗漏
            val_filenames.discard(file_name)

    print('\n' + '-'*30)
    print(f'Processing Done!')
    print(f'Successfully moved: {processed_count} images.')
    
    if missing_labels > 0:
        print(f'Missing labels: {missing_labels}')
    
    if len(val_filenames) > 0:
        print(f'Warning: {len(val_filenames)} images listed in val.txt were NOT found in the source folder.')
        # print(val_filenames) # 如果需要调试可以打印出来
    else:
        print('All images in val.txt were found and processed.')

if __name__ == '__main__':
    main()
