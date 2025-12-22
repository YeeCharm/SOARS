import argparse
import json
import math
import os
import requests
import timm
import torch
import torchvision.transforms as T
import sys

from io import BytesIO
# 复用原有的 hooks 和 utils，确保你之前的辅助文件路径正确
from src.hooks import get_self_attention, process_self_attention, get_second_last_out, feats
from src.webdatasets_util import cc2coco_format, create_webdataset_tar, read_coco_format_wds
from PIL import Image
# [关键修复] 解除像素限制，防止读取超大卫星图时报错
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration, AddedToken

def generate_caption(model, processor, images, prompt="a photography of"):
    image_token = AddedToken("<image>", normalized=False, special=True)
    processor.tokenizer.add_tokens([image_token], special_tokens=True)

    model.resize_token_embeddings(len(processor.tokenizer), pad_to_multiple_of=64) 
    model.config.image_token_index = len(processor.tokenizer) - 1
    inputs = processor(images=images, text=[prompt] * len(images), return_tensors="pt").to(next(model.parameters()).device)
    inputs['pixel_values'] = inputs['pixel_values'].float()

    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return [x.strip() for x in generated_text]    

def run_extraction(model_name, data_dir, ann_path, batch_size, resize_dim=518, crop_dim=518, out_path=None, 
                   write_as_wds=False, num_shards=25, n_in_splits=4, in_batch_offset=0, out_offset=0,
                   extract_cls=False, extract_avg_self_attn=False, extract_second_last_out=False,
                   extract_patch_tokens=False, extract_self_attn_maps=False, extract_disentangled_self_attn=False, 
                   blip_model_name=None):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Loading model: {model_name}")

    # ==========================
    # 1. 模型加载与配置自动探测
    # ==========================
    # 使用 timm 加载模型，启用 dynamic_img_size 以适配遥感多尺度
    try:
        model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,  # 移除分类头
            dynamic_img_size=True 
        )
    except Exception as e:
        print(f"Error loading model {model_name} from timm: {e}")
        print("Please ensure you have the latest timm installed: pip install --upgrade timm")
        sys.exit(1)

    # 获取数据配置 (Mean/Std/Interpolation)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    
    # 强制覆盖 Resize/Crop (如果用户指定)
    if resize_dim is not None and crop_dim is not None:
        transforms.transforms[0] = T.Resize(resize_dim, interpolation=T.InterpolationMode.BICUBIC)
        transforms.transforms[1] = T.CenterCrop(crop_dim)
    
    print(f"Transforms: {transforms}")

    model.eval()
    model.to(device)

    # ==========================
    # 2. 自动探测 Patch Size 和 Global Tokens
    # ==========================
    # DINOv3 (SAT) 可能有 5 个 global tokens (1 CLS + 4 REG)，也可能不同
    # 我们通过一次 Dummy Forward 来自动计算，而不是硬编码
    with torch.no_grad():
        dummy_input = torch.zeros(1, 3, crop_dim, crop_dim).to(device)
        dummy_out = model.forward_features(dummy_input) # (B, N_all, D)
        
        # 获取 Embedding Dimension
        embed_dim = dummy_out.shape[-1]
        
        # 获取 Patch Size (尝试从配置读取，否则默认 16 或 14)
        if hasattr(model, 'patch_embed'):
            patch_size = model.patch_embed.patch_size
            if isinstance(patch_size, tuple): patch_size = patch_size[0]
        else:
            # 简单的名称推断 fallback
            patch_size = 16 if 'patch16' in model_name else 14
        
        # 计算理论上的 Patch Token 数量
        grid_h = crop_dim // patch_size
        grid_w = crop_dim // patch_size
        num_expected_patch_tokens = grid_h * grid_w
        
        # 实际输出的总 Token 数
        total_tokens_out = dummy_out.shape[1]
        
        # 推断 Global Tokens (CLS + Registers)
        num_global_tokens = total_tokens_out - num_expected_patch_tokens
        
        # 获取 Head 数量
        num_attn_heads = model.blocks[-1].attn.num_heads
        
        # Scale for attention (通常是 1 / sqrt(head_dim))
        head_dim = embed_dim // num_attn_heads
        scale = head_dim ** -0.5

    print(f"Auto-detected configuration:")
    print(f"  - Embed Dim: {embed_dim}")
    print(f"  - Patch Size: {patch_size}")
    print(f"  - Total Tokens: {total_tokens_out}")
    print(f"  - Expected Patch Tokens ({grid_h}x{grid_w}): {num_expected_patch_tokens}")
    print(f"  - Detected Global Tokens (CLS+Reg): {num_global_tokens}")
    print(f"  - Num Heads: {num_attn_heads}")
    print(f"  - Attention Scale: {scale}")

    # ==========================
    # 3. 注册 Hooks
    # ==========================
    if extract_second_last_out:
        model.blocks[-2].register_forward_hook(get_second_last_out)
    
    # 注册 Attention Hook
    if extract_avg_self_attn or extract_self_attn_maps or extract_disentangled_self_attn:
        # timm 的 ViT 结构通常是 model.blocks[i].attn.qkv
        model.blocks[-1].attn.qkv.register_forward_hook(get_self_attention)

    # BLIP 模型加载 (如果有)
    if blip_model_name is not None:
        blip_processor = Blip2Processor.from_pretrained(blip_model_name)
        blip_model = Blip2ForConditionalGeneration.from_pretrained(
            blip_model_name, torch_dtype=torch.float16
        ).to(device)
        blip_processor.num_query_tokens = blip_model.config.num_query_tokens

    # ==========================
    # 4. 数据集加载
    # ==========================
    if os.path.isdir(ann_path):
        data = cc2coco_format(ann_path, n_in_splits, in_batch_offset)
    elif '.tar' in ann_path:
        data = read_coco_format_wds(ann_path)
    else:
        if ann_path.endswith('.json'):
            print("Loading annotations JSON...")
            with open(ann_path, 'r') as f:
                data = json.load(f)
        else:
            print("Loading annotations PTH...")
            data = torch.load(ann_path, weights_only=False)

    print(f"Starting feature extraction for {len(data['images'])} images...")
    
    # ==========================
    # 5. 提取循环
    # ==========================
    n_imgs = len(data['images'])
    n_batch = math.ceil(n_imgs / batch_size)
    n_errors = 0

    for i in tqdm(range(n_batch)):
        start = i * batch_size
        end = start + batch_size if i < n_batch - 1 else n_imgs
        current_batch_size = end - start
        
        raw_imgs = []
        failed_ids = []
        
        for j in range(start, end):
            # 图片加载逻辑 (复用之前的强壮逻辑)
            img_info = data['images'][j]
            pil_img = None
            
            # 尝试多种路径策略
            paths_to_try = []
            if 'file_name' in img_info: paths_to_try.append(img_info['file_name'])
            if 'coco_url' in img_info: paths_to_try.append(img_info['coco_url'])
            
            # 构建潜在路径列表
            candidate_paths = []
            for p in paths_to_try:
                if os.path.isabs(p): candidate_paths.append(p)
                if data_dir: candidate_paths.append(os.path.join(data_dir, p))
                # 兼容性路径
                if data_dir and 'train' in p: candidate_paths.append(os.path.join(data_dir, f"train2014/{os.path.basename(p)}"))
            
            for p in candidate_paths:
                if os.path.exists(p):
                    try:
                        pil_img = Image.open(p).convert('RGB')
                        break
                    except: continue
            
            if pil_img is None:
                # Placeholder for failure
                pil_img = Image.new("RGB", (crop_dim, crop_dim))
                failed_ids.append(j)
                n_errors += 1
            
            raw_imgs.append(pil_img)

        # 预处理
        batch_imgs = torch.stack([transforms(img) for img in raw_imgs]).to(device)

        with torch.no_grad():
            # Forward Pass
            # timm return: (B, N, D) 包括 CLS, Reg, Patches
            features_all = model.forward_features(batch_imgs) 
            
            # 分离 tokens
            # Token 0 is CLS
            cls_tokens = features_all[:, 0, :]
            
            # Patch tokens starts after global tokens
            patch_tokens = features_all[:, num_global_tokens:, :]

            # 处理 Attention
            avg_self_attn_token = None
            self_attn_maps_out = None
            disentangled_self_attn = None
            
            if extract_avg_self_attn or extract_self_attn_maps or extract_disentangled_self_attn:
                # 使用 hook 抓取的 qkv 计算 attention
                # 注意：process_self_attention 需要传入 num_tokens (global + patch)
                # 计算逻辑在 src/hooks.py 中，我们需要确保传入正确的 num_global_tokens
                
                # 计算总 token 数 (用于 hook 内部 reshape)
                total_tokens_hook = num_global_tokens + patch_tokens.shape[1]
                
                self_attn_avg, self_attn_maps_raw = process_self_attention(
                    feats['self_attn'], 
                    current_batch_size, 
                    total_tokens_hook, 
                    num_attn_heads, 
                    embed_dim, 
                    scale, 
                    num_global_tokens, # 这里的 num_global_tokens 非常关键，用于切片掉 global tokens 仅保留 patch attention
                    ret_self_attn_maps=True
                )
                
                if extract_avg_self_attn:
                    # (B, N_patches) -> (B, N_patches, 1) * (B, N_patches, D) -> mean -> (B, D)
                    avg_self_attn_token = (self_attn_avg.unsqueeze(-1) * patch_tokens).mean(dim=1)
                
                if extract_disentangled_self_attn:
                    # (B, Heads, N_patches) -> softmax -> unsqueeze -> weighted mean
                    maps_soft = self_attn_maps_raw.softmax(dim=-1)
                    # (B, N_patches, 1, D) * (B, N_patches, Heads, 1) ?? 
                    # 按照原代码逻辑: (x.unsqueeze(1) * maps.unsqueeze(-1)).mean(2)
                    # x: (B, N_p, D), maps: (B, Heads, N_p)
                    # 应该: (B, 1, N_p, D) * (B, H, N_p, 1) -> (B, H, N_p, D) -> mean(dim=2) -> (B, H, D)
                    
                    # 修正维度操作以匹配原始逻辑
                    # original: (outs['x_norm_patchtokens'].unsqueeze(1) * self_attn_maps.unsqueeze(-1)).mean(dim=2)
                    # patch_tokens: [B, Np, D]
                    # maps_soft: [B, H, Np] -> permute to [B, Np, H] ?? No, map is usually [B, H, Np]
                    
                    # 让我们看看 DINOv2 extraction 里的逻辑：
                    # disentangled_self_attn = (outs['x_norm_patchtokens'].unsqueeze(1) * self_attn_maps.unsqueeze(-1)).mean(dim=2)
                    # 这里假设 self_attn_maps 是 [B, Heads, N_patches] (由 process_self_attention 返回)
                    # 那么 self_attn_maps.unsqueeze(-1) 是 [B, Heads, N_patches, 1]
                    # x_norm_patchtokens.unsqueeze(1) 是 [B, 1, N_patches, D]
                    # 乘法结果: [B, Heads, N_patches, D]
                    # mean(dim=2) -> [B, Heads, D] -> 这是我们想要的每个 Head 的特征向量
                    
                    disentangled_self_attn = (patch_tokens.unsqueeze(1) * maps_soft.unsqueeze(-1)).mean(dim=2)

                if extract_self_attn_maps:
                    self_attn_maps_out = self_attn_maps_raw

            if extract_second_last_out:
                second_last_cls = feats['second_last_out'][:, 0, :]
                
            if blip_model_name is not None:
                new_capts = generate_caption(blip_model, blip_processor, raw_imgs)

        # ==========================
        # 6. 保存结果
        # ==========================
        for j in range(start, end):
            if j in failed_ids: continue
            
            idx = j - start
            # 始终保存 CLS，除非用户显式不想要（但在 Talk2DINO 中通常作为基础特征）
            data['images'][j]['dino_features'] = cls_tokens[idx].cpu()
            
            if extract_avg_self_attn and avg_self_attn_token is not None:
                data['images'][j]['avg_self_attn_out'] = avg_self_attn_token[idx].cpu()
            
            if extract_disentangled_self_attn and disentangled_self_attn is not None:
                data['images'][j]['disentangled_self_attn'] = disentangled_self_attn[idx].cpu()
                
            if extract_self_attn_maps and self_attn_maps_out is not None:
                data['images'][j]['self_attn_maps'] = self_attn_maps_out[idx].cpu()

            if extract_patch_tokens:
                data['images'][j]['patch_tokens'] = patch_tokens[idx].cpu()

            if extract_second_last_out:
                data['images'][j]['second_last_out'] = second_last_cls[idx].cpu()
                
            if blip_model_name is not None:
                data['annotations'][j]['caption'] = new_capts[idx]

    print(f"Extraction finished. Failed images: {n_errors}")
    
    if write_as_wds:
        os.makedirs(out_path, exist_ok=True)
        create_webdataset_tar(data, out_path, num_shards, out_offset)
    else:
        if out_path is None: out_path = os.path.splitext(ann_path)[0] + '.pth'
        torch.save(data, out_path)
        print(f"Features saved to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract DINOv3 Features using timm")
    parser.add_argument('--ann_path', type=str, required=True, help="Path to annotation file (json or pth)") 
    parser.add_argument('--data_dir', type=str, required=True, help="Root directory of images") 
    parser.add_argument('--out_path', type=str, default=None, help="Output path")
    
    # 默认模型改为你需要的 DINOv3 (SAT)
    parser.add_argument('--model', type=str, default="timm/vit_large_patch16_dinov3.sat493m", help="timm model name")
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--resize_dim', type=int, default=518, help="Resize dimension (important for RS small objects)")
    parser.add_argument('--crop_dim', type=int, default=518, help="Crop dimension")
    
    parser.add_argument('--extract_cls', action="store_true")
    parser.add_argument('--extract_avg_self_attn', action="store_true")
    parser.add_argument('--extract_disentangled_self_attn', action="store_true")
    parser.add_argument('--extract_patch_tokens', action="store_true")
    parser.add_argument('--extract_self_attn_maps', action="store_true")
    parser.add_argument('--extract_second_last_out', action="store_true")
    
    parser.add_argument('--blip_model', type=str, default=None)
    parser.add_argument('--write_as_wds', action="store_true")
    parser.add_argument('--n_shards', type=int, default=25)
    parser.add_argument('--n_in_split', type=int, default=1)
    parser.add_argument('--in_batch_offset', type=int, default=0)
    parser.add_argument('--out_offset', type=int, default=0)

    args = parser.parse_args()
    
    run_extraction(
        args.model, args.data_dir, args.ann_path, args.batch_size, 
        args.resize_dim, args.crop_dim, args.out_path,
        args.write_as_wds, args.n_shards, args.n_in_split, args.in_batch_offset, args.out_offset,
        args.extract_cls, args.extract_avg_self_attn, args.extract_second_last_out, 
        args.extract_patch_tokens, args.extract_self_attn_maps,
        args.extract_disentangled_self_attn, args.blip_model
    )

if __name__ == '__main__':
    main()