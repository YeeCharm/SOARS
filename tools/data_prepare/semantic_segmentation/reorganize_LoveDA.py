import argparse
import numpy as np
import mmcv
import os
import os.path as osp
from pathlib import Path
from mmengine.utils import ProgressBar, mkdir_or_exist
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Convert multi-class labels to colored labels')
    parser.add_argument('image_path', type=str, help='Path to the label image or directory')
    parser.add_argument('-o', '--output_dir', type=str, default=None, help='Output directory (default: same as input)')
    parser.add_argument('--colors', type=str, nargs='+', default=None, 
                       help='Colors for each class in format "class_id:R,G,B". Example: "2:128,128,128" "3:255,0,0"')
    parser.add_argument('--rgb_dir', type=str, default=None,
                       help='Directory containing RGB images (required for overlay)')
    parser.add_argument('--alpha', type=float, default=0.6,
                       help='Alpha value for mask overlay (0.0-1.0, default: 0.6)')
    parser.add_argument('--bg_class_id', type=int, default=1,
                       help='Class ID to use as background (default: 1)')
    return parser.parse_args()


def convert_multi_class_to_color(image_path, color_map, rgb_dir=None, alpha=0.6, bg_class_id=1):
    """
    Convert multi-class label image to colored label image using demo.py style overlay.
    
    Args:
        image_path: Path to the label image
        color_map: Dictionary mapping class IDs to RGB colors
        rgb_dir: Directory containing RGB images (required)
        alpha: Alpha value for mask overlay (0.0-1.0)
        bg_class_id: Class ID to use as background
    
    Returns:
        Colored label image (H, W, 3)
    """
    label = mmcv.imread(image_path, 'grayscale')
    h, w = label.shape
    
    # Load RGB image
    if rgb_dir is None:
        raise ValueError("rgb_dir is required for overlay visualization")
    
    base_name = osp.basename(image_path)
    rgb_path = osp.join(rgb_dir, base_name)
    
    if not osp.exists(rgb_path):
        raise ValueError(f"RGB image not found: {rgb_path}")
    
    rgb_image = mmcv.imread(rgb_path, 'color')
    rgb_image = rgb_image[:h, :w, :]
    
    # Create qualitative plot (mask overlay) - same as demo.py
    qualitative_plot = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id in np.unique(label):
        if class_id == bg_class_id:
            # Background remains black (0, 0, 0)
            continue
        if class_id in color_map:
            qualitative_plot[label == class_id] = np.array(color_map[class_id])
    
    # Convert to matplotlib format and overlay - same as demo.py
    fig, ax = plt.subplots(figsize=(w/100, h/100), dpi=100)
    ax.axis('off')
    ax.imshow(rgb_image)
    ax.imshow(qualitative_plot, alpha=alpha)
    plt.tight_layout(pad=0)
    
    # Convert to numpy array
    fig.canvas.draw()
    result = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    result = result.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    plt.close(fig)
    
    return result


def main():
    args = parse_args()
    
    input_path = args.image_path
    output_dir = args.output_dir
    rgb_dir = args.rgb_dir
    alpha = args.alpha
    bg_class_id = args.bg_class_id
    
    # Define color map matching demo.py
    # Class IDs: 1=background, 2=building, 3=road, 4=water, 5=barren, 6=forest, 7=agricultural
    if args.colors:
        color_map = {}
        for color_spec in args.colors:
            class_id_str, rgb_str = color_spec.split(':')
            class_id = int(class_id_str)
            rgb = tuple(map(int, rgb_str.split(',')))
            color_map[class_id] = rgb
    else:
        # Default colors matching demo.py palette
        color_map = {
            2: (200, 100, 100),  # building - red tone
            3: (100, 200, 100),  # road - green tone
            4: (100, 100, 200),  # water - blue tone
            5: (200, 200, 100),  # barren - yellow tone
            6: (200, 100, 200),  # forest - magenta tone
            7: (100, 200, 200)   # agricultural - cyan tone
        }
    
    if rgb_dir is None:
        print("Error: --rgb_dir is required for overlay visualization")
        return
    
    if output_dir is None:
        if osp.isfile(input_path):
            output_dir = osp.dirname(input_path)
        else:
            output_dir = input_path + '_colored'
    
    mkdir_or_exist(output_dir)
    
    if osp.isfile(input_path):
        image_list = [input_path]
    else:
        image_list = list(Path(input_path).glob('*.png')) + list(Path(input_path).glob('*.jpg'))
    
    if len(image_list) == 0:
        print(f'No images found in {input_path}')
        return
    
    print(f'Found {len(image_list)} image(s) to convert')
    print(f'Color map: {color_map}')
    print(f'RGB directory: {rgb_dir}')
    print(f'Alpha: {alpha}')
    print(f'Background class ID: {bg_class_id}')
    print(f'Output directory: {output_dir}')
    
    prog_bar = ProgressBar(len(image_list))
    
    for image_path in image_list:
        base_name = osp.basename(str(image_path))
        save_path = osp.join(output_dir, base_name)
        
        try:
            colored_image = convert_multi_class_to_color(
                str(image_path), color_map, rgb_dir, alpha, bg_class_id)
            
            mmcv.imwrite(colored_image, save_path)
        except Exception as e:
            print(f'\nError processing {image_path}: {e}')
        
        prog_bar.update()
    
    print('\nDone!')


if __name__ == '__main__':
    main()
