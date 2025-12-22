import json
import os

def update_coco_urls(input_path, output_path, new_image_root):
    """
    读取 JSON 文件，修改 coco_url 路径，并保存到指定的新文件中。
    """
    # 1. 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"错误：找不到输入文件 {input_path}")
        return

    print(f"正在读取文件：{input_path} ...")
    
    try:
        # 2. 读取数据
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'images' not in data:
            print("错误：JSON 数据中缺少 'images' 字段。")
            return
            
        count = 0
        # 3. 修改 coco_url
        for img in data['images']:
            # 获取原始文件名 (例如: airport_1.jpg)
            if 'file_name' in img:
                file_name = img['file_name']
                
                # 拼接新的完整路径
                # rstrip('/') 确保路径末尾没有多余的斜杠，然后加上 / 和文件名
                new_url = f"{new_image_root.rstrip('/')}/{file_name}"
                
                # 更新字段
                img['coco_url'] = new_url
                count += 1
        
        print(f"已处理 {count} 条图片数据。")
        
        # 4. 确保输出目录存在（如果不存在则创建）
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"已创建输出目录：{output_dir}")

        # 5. 保存到新文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        print(f"成功！")
        print(f"新文件已保存为：{output_path}")
        
        # 打印示例以供检查
        if data['images']:
            print("\n--- 修改后数据示例 (ID: 0) ---")
            print(f"Old File Name: {data['images'][0]['file_name']}")
            print(f"New COCO URL : {data['images'][0]['coco_url']}")

    except json.JSONDecodeError:
        print("错误：JSON 文件格式不正确。")
    except Exception as e:
        print(f"发生未知错误：{e}")

if __name__ == "__main__":
    # --- 配置路径 ---
    
    # 1. 原始 JSON 文件路径
    input_json = "/home/luoyichang/code/Talk2DINO/data/RSICD/annotations/rsicd_coco_format.json"
    
    # 2. 新的保存路径 (文件名已改为 rsicd_coco_format_modify.json)
    output_json = "/home/luoyichang/code/Talk2DINO/data/RSICD/annotations/rsicd_coco_format_modify.json"
    
    # 3. 图片文件夹的新路径前缀
    new_image_path_prefix = "/home/luoyichang/code/Talk2DINO/data/RSICD/images"
    
    # --- 运行 ---
    update_coco_urls(input_json, output_json, new_image_path_prefix)