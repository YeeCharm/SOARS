import argparse
import clip
import json
import math
import glob  # [新增] 用于文件搜索
import os
import torch
import torchvision.transforms as T

from src.hooks import get_self_attention, process_self_attention, get_second_last_out, feats, get_clip_second_last_dense_out
from PIL import Image
from tqdm import tqdm
from transformers import BertModel, AutoTokenizer
from src.webdatasets_util import cc2coco_format, create_webdataset_tar
from src.hooks import get_all_out_tokens, feats
    

def run_bert_extraction(model_name, ann_path, batch_size, out_path, extract_dense_out=False, extract_second_last_dense_out=False,
                          write_as_wds=False, num_shards=25, n_in_splits=4, in_batch_offset=0, out_offset=0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # [优化] 如果 out_path 未指定，默认覆盖原文件
    if out_path is None:
        out_path = ann_path

    print(f"[-] Loading model: {model_name}...")
    if 'bert' in model_name:
        model_type = 'bert'
        field_name = 'bert-base_features'
        model = BertModel.from_pretrained(model_name, output_hidden_states = False)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        model_type = 'clip'
        field_name = 'ann_feats'
        model, _ = clip.load(model_name, device=device)
        if extract_dense_out:
            model.ln_final.register_forward_hook(get_all_out_tokens)
        if extract_second_last_dense_out:
            model.transformer.resblocks[-2].register_forward_hook(get_clip_second_last_dense_out)
            
    model.eval()
    model.to(device)
    
    print(f"[-] Loading data from {ann_path}...")
    data = torch.load(ann_path) # .pth file
    
    # 兼容两种格式：如果是 dict 则取 'annotations'，否则直接是 list
    if isinstance(data, dict):
        annotations = data['annotations']
    else:
        annotations = data
        
    print(f"[-] Extracting text features for {len(annotations)} captions...")
    
    bs = batch_size
    n_batches = math.ceil(len(annotations) / bs)
    
    for i in tqdm(range(n_batches)):
        batch = annotations[i*bs : (i+1)*bs]
        captions = [b['caption'] for b in batch]
        
        with torch.no_grad():
            if model_type == 'clip':
                text = clip.tokenize(captions, truncate=True).to(device)
                text_features = model.encode_text(text)
                
                if extract_dense_out:
                    text_features = feats['clip_txt_out_tokens']
                if extract_second_last_dense_out:
                    text_features = feats['clip_second_last_out']
                    
            elif model_type == 'bert':
                inputs = tokenizer(captions, padding=True, truncation=True, return_tensors="pt").to(device)
                outputs = model(**inputs)
                text_features = outputs.last_hidden_state[:, 0, :] # CLS token

            # 保存回 data 结构中
            for idx, feature in enumerate(text_features):
                batch[idx][field_name] = feature.cpu() # detach from gpu
    
    # 保存结果
    print(f"[-] Saving updated features to {out_path}...")
    torch.save(data, out_path)
    print("[-] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_path', type=str, default=None, required=True, help="Input annotation path (or prefix)")
    parser.add_argument('--model', type=str, default="ViT-B/32")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--out_path', type=str, default=None, help="Output path (optional)") 
    parser.add_argument('--extract_dense_out', action="store_true")
    parser.add_argument('--extract_second_last_dense_out', action="store_true")
    # 以下参数暂时保留以兼容接口，但在这个简化脚本中主要用于标准 pth 处理
    parser.add_argument('--write_as_wds', action="store_true") 
    parser.add_argument('--n_shards', type=int, default=10)
    parser.add_argument('--n_in_splits', type=int, default=1)
    parser.add_argument('--in_batch_offset', type=int, default=0)
    parser.add_argument('--out_offset', type=int, default=0)
    args = parser.parse_args()
    
    # ================= 智能文件列表生成逻辑 =================
    target_files = []
    
    # 情况 1: 输入的文件直接存在 (单文件模式)
    if os.path.exists(args.ann_path):
        target_files.append(args.ann_path)
    
    # 情况 2: 输入的文件不存在，尝试作为前缀搜索 (分片模式)
    else:
        # 去掉 .pth 后缀 (如果有)
        base_prefix = os.path.splitext(args.ann_path)[0]
        # 搜索模式：前缀 + _part*.pth
        search_pattern = f"{base_prefix}_part*.pth"
        found_files = sorted(glob.glob(search_pattern))
        
        if found_files:
            print(f"[√] 自动识别为分片模式，找到 {len(found_files)} 个文件：")
            for f in found_files:
                print(f"    - {f}")
            target_files = found_files
        else:
            print(f"[!] 错误: 找不到文件 {args.ann_path}，也找不到匹配 {search_pattern} 的分片文件。")
            exit(1)
            
    # ================= 循环处理所有文件 =================
    print(f"\n[-] 开始批量提取文本特征 (共 {len(target_files)} 个任务)...")
    
    for idx, input_file in enumerate(target_files):
        print(f"\n>>> 处理进度 [{idx+1}/{len(target_files)}]: {input_file}")
        
        # 自动推导输出路径
        # 如果用户没指定 out_path，默认覆盖原文件 (None)
        # 如果用户指定了 out_path (例如 out.pth)，且是多文件模式，则自动加上 _partX
        current_out_path = None
        if args.out_path is not None:
            if len(target_files) > 1:
                # 保持 part 后缀
                base_name = os.path.basename(input_file)
                # 简单的逻辑：输出到 args.out_path 的目录，文件名保持输入的文件名
                out_dir = os.path.dirname(args.out_path)
                current_out_path = os.path.join(out_dir, base_name)
            else:
                current_out_path = args.out_path

        run_bert_extraction(
            args.model, 
            input_file, 
            args.batch_size, 
            current_out_path, 
            args.extract_dense_out, 
            args.extract_second_last_dense_out,
            args.write_as_wds, args.n_shards, args.n_in_splits, 
            args.in_batch_offset, args.out_offset
        )
        
    print("\n[√] 所有文本特征提取完成！")