import argparse
import json
import math
import os
import timm
import torch
import torchvision.transforms as T
import sys
from PIL import Image
from tqdm import tqdm

# [关键设置] 解除像素限制
Image.MAX_IMAGE_PIXELS = None

# 复用原有的 hooks 和 utils
from src.hooks import get_self_attention, process_self_attention, feats

# 定义我们要提取的层数 (ViT-Large 0-23)
# 11: 中层 (细节/纹理)
# 17: 中深层 (结构)
# 23: 深层 (语义/最后一层)
TARGET_LAYERS = [11, 17, 23]

def run_extraction(model_name, data_dir, ann_path, batch_size, resize_dim, crop_dim, out_path, 
                   extract_avg_self_attn=False, extract_disentangled_self_attn=False):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. 加载模型
    print(f"Loading model: {model_name}")
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=0,
        dynamic_img_size=True
    )
    
    # 2. 注册 Hooks 以获取中间层特征
    # 我们使用一个字典来存储每一层的输出
    layer_outputs = {}
    
    def get_layer_hook(layer_id):
        def hook(module, input, output):
            # output 是 (Batch, Tokens, Dim)
            layer_outputs[layer_id] = output
        return hook

    # 遍历模型 Block 注册 Hook
    for i, block in enumerate(model.blocks):
        if i in TARGET_LAYERS:
            block.register_forward_hook(get_layer_hook(i))
            print(f"Registered hook for Layer {i}")

    # 注册 Attention Hook (只获取最后一层的 Attention Map 用于聚合)
    # 注意：我们假设最后一层(23)的 Attention 最能代表物体位置
    model.blocks[-1].attn.qkv.register_forward_hook(get_self_attention)

    # 3. 数据预处理
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    if resize_dim is not None and crop_dim is not None:
        transforms.transforms[0] = T.Resize(resize_dim, interpolation=T.InterpolationMode.BICUBIC)
        transforms.transforms[1] = T.CenterCrop(crop_dim)
    
    model.eval()
    model.to(device)

    # 4. 自动探测参数
    with torch.no_grad():
        dummy_input = torch.zeros(1, 3, crop_dim, crop_dim).to(device)
        _ = model(dummy_input) # 跑一次以触发 hook 探测维度
        
        # 从最后一层获取维度信息
        last_feat = layer_outputs[TARGET_LAYERS[-1]]
        embed_dim = last_feat.shape[-1]
        total_tokens = last_feat.shape[1]
        
        # 计算 Patch 和 Global tokens
        if hasattr(model, 'patch_embed'):
            patch_size = model.patch_embed.patch_size
            if isinstance(patch_size, tuple): patch_size = patch_size[0]
        else:
            patch_size = 16
            
        grid_h = crop_dim // patch_size
        grid_w = crop_dim // patch_size
        num_patch_tokens = grid_h * grid_w
        num_global_tokens = total_tokens - num_patch_tokens
        
        num_attn_heads = model.blocks[-1].attn.num_heads
        scale = (embed_dim // num_attn_heads) ** -0.5
        
        print(f"Auto-detected: Embed Dim={embed_dim}, Heads={num_attn_heads}, Global Tokens={num_global_tokens}")

    # 5. 加载数据
    if ann_path.endswith('.json'):
        with open(ann_path, 'r') as f:
            data = json.load(f)
        images = data['images']
        annotations = data['annotations']
        categories = data.get('categories', [])
    else:
        # 兼容 .pth 读取
        data = torch.load(ann_path)
        images = data['images']
        annotations = data['annotations']
        categories = data.get('categories', [])

    # 结果容器
    final_images = []
    
    # 6. 开始提取
    print(f"Starting multi-scale extraction for {len(images)} images...")
    
    for i in tqdm(range(0, len(images), batch_size)):
        batch_images_meta = images[i : i + batch_size]
        batch_tensors = []
        valid_indices = [] # 记录这一批次里成功的图片索引
        
        # 读取图片
        for idx, img_meta in enumerate(batch_images_meta):
            try:
                img_path = img_meta['file_name']
                if data_dir and not os.path.isabs(img_path):
                    img_path = os.path.join(data_dir, img_path)
                
                pil_img = Image.open(img_path).convert('RGB')
                batch_tensors.append(transforms(pil_img))
                valid_indices.append(idx)
            except Exception as e:
                # print(f"Error reading {img_meta.get('file_name')}: {e}")
                # 失败则填黑图占位，保证 batch 维度对齐
                batch_tensors.append(torch.zeros(3, crop_dim, crop_dim))
                valid_indices.append(idx)

        if not batch_tensors:
            continue
            
        batch_input = torch.stack(batch_tensors).to(device)
        
        with torch.no_grad():
            # 前向传播 (Hooks 会自动捕获 11, 17, 23 层特征)
            _ = model(batch_input)
            
            # --- 核心逻辑：多尺度特征融合 ---
            
            # 1. 获取最后一层的 Attention Map (用于聚合所有层)
            # 使用 hooks.py 中的逻辑计算
            # feats['self_attn'] 是最后一层的 qkv 乘积
            self_attn_avg, self_attn_maps_raw = process_self_attention(
                feats['self_attn'], 
                len(batch_tensors), 
                total_tokens, 
                num_attn_heads, 
                embed_dim, 
                scale, 
                num_global_tokens, 
                ret_self_attn_maps=True
            )
            
            maps_soft = self_attn_maps_raw.softmax(dim=-1) # (B, Heads, N_patches)
            
            # 2. 遍历每一层，应用 Attention 聚合
            multi_scale_avg_list = []
            multi_scale_disentangled_list = []
            
            for layer_idx in TARGET_LAYERS:
                # 获取该层特征 (B, Total_Tokens, C)
                layer_feat = layer_outputs[layer_idx]
                
                # 剥离 Patch Tokens (B, N_patches, C)
                patch_tokens = layer_feat[:, num_global_tokens:, :]
                
                # 聚合方式 A: Avg Self Attn (B, C)
                if extract_avg_self_attn:
                    # (B, N, 1) * (B, N, C) -> (B, C)
                    feat_avg = (self_attn_avg.unsqueeze(-1) * patch_tokens).mean(dim=1)
                    multi_scale_avg_list.append(feat_avg)
                
                # 聚合方式 B: Disentangled Self Attn (B, Heads, C)
                if extract_disentangled_self_attn:
                    # (B, 1, N, C) * (B, H, N, 1) -> (B, H, N, C) -> mean -> (B, H, C)
                    feat_disentangled = (patch_tokens.unsqueeze(1) * maps_soft.unsqueeze(-1)).mean(dim=2)
                    multi_scale_disentangled_list.append(feat_disentangled)

            # 3. 拼接 (Concatenate)
            # 维度从 1024 变为 3072 (1024*3)
            if extract_avg_self_attn:
                final_avg_feat = torch.cat(multi_scale_avg_list, dim=-1) # (B, 3072)
                
            if extract_disentangled_self_attn:
                final_disentangled_feat = torch.cat(multi_scale_disentangled_list, dim=-1) # (B, Heads, 3072)

        # 保存结果
        for idx in valid_indices:
            # 获取对应的 meta 信息并深拷贝，防止污染
            img_meta = batch_images_meta[idx].copy()
            
            # 保存多尺度特征
            if extract_avg_self_attn:
                # 使用原来的 key 名，但内容已经是拼接后的了
                # 这样 dataset.py 修改量最小，但要记得维度变了
                img_meta['avg_self_attn_out'] = final_avg_feat[idx].cpu()
                
            if extract_disentangled_self_attn:
                # 同样使用原 key 名
                img_meta['disentangled_self_attn'] = final_disentangled_feat[idx].cpu()
                
            # 也可以保留最后一层的 CLS token 用于备用
            img_meta['dino_features'] = layer_outputs[23][idx, 0, :].cpu()
            
            final_images.append(img_meta)

    # 7. 保存最终 .pth
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"Saving features to {out_path} ...")
    torch.save({
        'images': final_images,
        'annotations': annotations,
        'categories': categories
    }, out_path)
    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default="")
    parser.add_argument('--model', type=str, default="timm/vit_large_patch16_dinov3.lvd1689m")
    parser.add_argument('--resize_dim', type=int, default=512)
    parser.add_argument('--crop_dim', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--extract_avg_self_attn', action='store_true')
    parser.add_argument('--extract_disentangled_self_attn', action='store_true')
    
    args = parser.parse_args()
    
    run_extraction(
        args.model, args.data_dir, args.ann_path, args.batch_size, 
        args.resize_dim, args.crop_dim, args.out_path,
        args.extract_avg_self_attn, args.extract_disentangled_self_attn
    )