import itertools
import os
import pickle
from math import sqrt
import re
import yaml

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange
from transformers import BertModel, AutoTokenizer
import torchvision.transforms as T
import clip
import importlib
import torchvision.transforms.functional as TF 

from models.builder import MODELS
from models.dinotext.pamr import PAMR
from models.dinotext.masker import DINOTextMasker
import us
from datasets import get_template

from src.model import ProjectionLayer, VisualProjectionLayer, CLIPLastLayer, DoubleMLP
from src.loss import Contrastive
from src.hooks import average_text_tokens, get_vit_out, feats

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# [辅助类] 用于捕获多层特征
class LayerFeatHook:
    def __init__(self):
        self.feats = {}
    
    def get_layer_hook(self, layer_id):
        def hook(module, input, output):
            self.feats[layer_id] = output
        return hook
    
    def clear(self):
        self.feats = {}

@MODELS.register_module()
class DINOText(nn.Module):
    
    def get_self_attention(self, module, input, output):
        self.feats['self_attn'] = output
        
    def get_clip_second_last_dense_out(self, model: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
        self.feats['clip_second_last_out'] = output
        self.feats['clip_second_last_out'].to(dtype=torch.float32)
    
    def get_all_out_tokens(self, model: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
        self.feats['clip_txt_out_tokens'] = output
        
    def __init__(
            self, model_name, resize_dim, clip_model_name, proj_class, proj_name, proj_model, avg_self_attn_token=False, disentangled_self_attn_token=True, loss=None, pre_trained=True,
            unfreeze_last_text_layer=False, unfreeze_last_image_layer=False, is_eval=True, use_avg_text_token=False, keep_cls=False, keep_end_seq=False, with_bg_clean=False, 
            dino_embed_dim=1024, # [新增参数] 默认 1024，如果是 3072 则自动开启多尺度
            **kwargs
    ):
        super().__init__()
        self.feats = {}
        self.model_name = model_name
        
        # [关键逻辑] 自动判断是否启用多尺度
        self.dino_embed_dim = dino_embed_dim
        self.multiscale = (dino_embed_dim == 3072)
        self.target_layers = [11, 17, 23] if self.multiscale else []
        self.layer_hook_helper = LayerFeatHook()
        
        # Loading Model
        if 'dinov2' in model_name:
            self.model_family = 'facebookresearch/dinov2' if 'dinov2' in model_name else 'facebookresearch/dino:main'
            self.model = torch.hub.load(self.model_family, model_name)                
        elif 'mae' in model_name or 'sam' in model_name or 'clip' in model_name or 'dino' in model_name:
            self.model = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=0,
                img_size=resize_dim,
                dynamic_img_size=True # 允许动态尺寸
            )
            if 'sam' in model_name:
                self.model.blocks[-1].register_forward_hook(get_vit_out)
        else:
            raise Exception("Unknown ViT model")

        # [关键逻辑] 注册多层特征 Hook
        if self.multiscale:
            print(f"[DINOText] Detected dim=3072, enabling Multiscale Feature Extraction (Layers {self.target_layers})")
            for i, block in enumerate(self.model.blocks):
                if i in self.target_layers:
                    block.register_forward_hook(self.layer_hook_helper.get_layer_hook(i))
        else:
            print(f"[DINOText] Detected dim={dino_embed_dim}, using Single Scale (Last Layer)")

        mean = (0.485, 0.456, 0.406) if not 'clip' in model_name else (0.4815, 0.4578, 0.4082)
        std = (0.229, 0.224, 0.225) if not 'clip' in model_name else (0.2686, 0.2613, 0.2758)
        self.image_transforms = T.Compose([
            T.Resize((resize_dim, resize_dim)),
            lambda x: T.ToTensor()(x) if not isinstance(x, torch.Tensor) else x / 255.0,
            T.Normalize(mean, std),
        ])
        
        self.model.to(device)
        self.model.requires_grad_(False)
        
        # CLIP Model
        self.clip_model_name = clip_model_name
        if 'bert' in self.clip_model_name:
            self.clip_model = BertModel.from_pretrained(self.clip_model_name, output_hidden_states = False)
            self.tokenizer = AutoTokenizer.from_pretrained(self.clip_model_name)
        else:
            self.clip_model, _ = clip.load(clip_model_name, device=device)
        self.clip_model.eval()
        self.clip_model.requires_grad_(False)
        
        if unfreeze_last_text_layer:
            for param in self.clip_model.transformer.resblocks[-1].parameters():
                param.requires_grad = True
            for param in self.clip_model.ln_final.parameters():
                param.requires_grad = True
            self.clip_model.text_projection.requires_grad = True
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # Loading Projection Layer Config
        # 注意：这里需要注入 dino_embed_dim，防止 config 文件里写的还是旧的
        with open(os.path.join('configs', f"{proj_class}.yaml"), 'r') as config_file:
            config = yaml.safe_load(config_file)['model']
            config['dino_embed_dim'] = self.dino_embed_dim # 覆盖 config 里的维度
            
        ProjClass = getattr(importlib.import_module('src.model'), proj_model)
        self.proj = ProjClass.from_config(config)
        
        if type(self.proj) == CLIPLastLayer:
            self.clip_model.transformer.resblocks[-2].register_forward_hook(self.get_clip_second_last_dense_out)
        
        if pre_trained:
            print(f"[DINOText] Loading projection weights from: weights/{proj_name}.pth")
            self.proj.load_state_dict(torch.load(os.path.join("weights", f"{proj_name}.pth"), 'cpu'))
        self.proj.to(device)
        
        self.masker = DINOTextMasker(similarity_type="cosine")
        self.masker = self.masker.eval()
        self.pamr = None
        self.with_bg_clean = with_bg_clean    
                
        self.avg_self_attn_token = avg_self_attn_token
        self.disentangled_self_attn_token = disentangled_self_attn_token
        
        if self.avg_self_attn_token or self.disentangled_self_attn_token or is_eval:
            self.model.blocks[-1].attn.qkv.register_forward_hook(self.get_self_attention)
            self.num_global_tokens = 5 if 'reg' in model_name or 'dinov3' in model_name else 1
            if 'sam' in self.model_name:
                self.num_global_tokens = 0
            if 'dinov3' in self.model_name:
                if 'vit_base' in self.model_name:
                    self.num_attn_heads = 12
                elif 'vit_large' in self.model_name:
                    self.num_attn_heads = 16
                else:
                    raise Exception("Unknown dinov3 model")
            else:
                self.num_attn_heads = self.model.num_heads
            self.scale = 0.125
        
        self.use_avg_text_token = use_avg_text_token
        if self.use_avg_text_token:
            self.feats = {}
            self.clip_model.ln_final.register_forward_hook(self.get_all_out_tokens)
            self.keep_cls = keep_cls
            self.keep_end_seq = keep_end_seq        
            
    
    def process_self_attention(self, output, batch_size, num_tokens, num_attn_heads, embed_dim, scale, num_global_tokens, ret_self_attn_maps=False):
        qkv = output.reshape(batch_size, num_tokens, 3, num_attn_heads, embed_dim // num_attn_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)
        self_attn_maps = attn[:, : , 0, num_global_tokens:]
        self_attn = self_attn_maps.mean(dim=1)
        self_attn = self_attn.softmax(dim=-1)
        if ret_self_attn_maps:
            return self_attn, self_attn_maps
        else:
            return self_attn
    
    def encode_text(self, tokenized_texts):
        if type(self.proj) == CLIPLastLayer:
            self.clip_model.encode_text(tokenized_texts)
            x = self.feats['clip_second_last_out']
            x = x.to(dtype=torch.float32)
        else:
            x = self.clip_model.encode_text(tokenized_texts)
        return x
    
    def encode_image(self, images):
        batch_size, _, _, _ = images.shape
        self_attn_maps = None
        
        # 清空 hook 缓存
        if self.multiscale:
            self.layer_hook_helper.clear()

        # 运行模型
        x = self.model(images)
        
        # 1. 获取 Patch Tokens
        if 'dinov2' in self.model_name:
            patch_tokens = x['x_norm_patchtokens']
        elif 'dinov3' in self.model_name:
            # 兼容 timm 版本差异
            if isinstance(x, dict):
                patch_tokens = x['x_norm_patchtokens']
            else:
                patch_tokens = x[:, self.num_global_tokens:, :] # 手动切片
        else:
             patch_tokens = x[:, 1:, :]

        batch_size, num_tokens, _ = patch_tokens.shape
        total_tokens = num_tokens + self.num_global_tokens
        
        # 2. 计算 Attention Maps (用于加权)
        if self.avg_self_attn_token or self.disentangled_self_attn_token:
            # 最后一层的 embed_dim (1024)
            last_embed_dim = patch_tokens.shape[-1]
            self_attn, self_attn_maps = self.process_self_attention(
                self.feats['self_attn'], batch_size, total_tokens, 
                self.num_attn_heads, last_embed_dim, self.scale, self.num_global_tokens, 
                ret_self_attn_maps=True
            )
            maps_soft = self_attn_maps.softmax(dim=-1) # (B, Heads, Tokens)

        # 3. 特征融合逻辑 (单尺度 vs 多尺度)
        if self.multiscale:
            # === 多尺度模式 (3072维) ===
            multi_scale_feats = []
            
            for layer_idx in self.target_layers:
                # 获取该层的输出
                layer_out = self.layer_hook_helper.feats[layer_idx] # (B, Total_Tokens, 1024)
                
                # 去掉 Global Tokens (CLS/Reg)
                layer_patches = layer_out[:, self.num_global_tokens:, :] 
                
                if self.avg_self_attn_token:
                    # 使用最后一层的 Attention 来加权每一层
                    feat = (self_attn.unsqueeze(-1) * layer_patches).mean(dim=1)
                elif self.disentangled_self_attn_token:
                    feat = (layer_patches.unsqueeze(1) * maps_soft.unsqueeze(-1)).mean(dim=2)
                else:
                    feat = layer_patches.mean(dim=1) # Fallback
                
                multi_scale_feats.append(feat)
            
            # 拼接: (B, 3072) 或 (B, Heads, 3072)
            x_final = torch.cat(multi_scale_feats, dim=-1)
            
        else:
            # === 单尺度模式 (1024维，旧逻辑) ===
            if self.avg_self_attn_token:
                x_final = (self_attn.unsqueeze(-1) * patch_tokens).mean(dim=1)
            elif self.disentangled_self_attn_token:
                x_final = (patch_tokens.unsqueeze(1) * maps_soft.unsqueeze(-1)).mean(dim=2)
            else:
                x_final = patch_tokens.mean(dim=1)

        return x_final, self_attn_maps

    def forward(self, image, text, return_logit_scale=False):
        with torch.no_grad():
            txt_embed = self.encode_text(text)
            
        img_embed, self_attn_maps = self.encode_image(image)
        
        if type(self.proj) == CLIPLastLayer:
            img_embed, txt_embed = self.proj(img_embed, txt_embed, ret_embeds=True, self_attn_maps=self_attn_maps, text_argmax=text.argmax(dim=-1))
        else:
            img_embed, txt_embed = self.proj(img_embed, txt_embed, ret_embeds=True, self_attn_maps=self_attn_maps)
        
        if return_logit_scale:
            return txt_embed, img_embed, self.logit_scale

        return txt_embed, img_embed
        
    def compute_loss(self, image, text, cosine=True, ret_similarity_matrix=True):
        ret = {}
        # 注意：这里假设外部调用已经做好了 encode，直接传 embedding 进来
        # 如果传的是 raw image/text，需要调 forward
        # 为了兼容性，这里暂时保持现状，但注意维度
        return ret # 实际上评估时不跑这个 loss

    # ================= TTA 模块 (保留你的增强逻辑) =================
    @torch.no_grad()
    def generate_masks(
            self, image, img_metas, text_emb, classnames, text_is_token=False, apply_pamr=False, background_func="weighted_average_sigmoid", lambda_bg=0.2,
    ):
        angles = [0, 90, 180, 270] 
        accum_mask = None
        accum_simmap = None
        
        for angle in angles:
            if angle == 0:
                img_input = image
            else:
                img_input = TF.rotate(image, angle)
            
            mask_i, simmap_i = self._inference_single(
                img_input, img_metas, text_emb, classnames, text_is_token, apply_pamr, background_func, lambda_bg
            )
            
            if angle != 0:
                mask_i = TF.rotate(mask_i, -angle)
                simmap_i = TF.rotate(simmap_i, -angle)
            
            if accum_mask is None:
                accum_mask = mask_i
                accum_simmap = simmap_i
            else:
                accum_mask += mask_i
                accum_simmap += simmap_i
        
        final_mask = accum_mask / len(angles)
        final_simmap = accum_simmap / len(angles)
        return final_mask, final_simmap

    def _inference_single(
            self, image, img_metas, text_emb, classnames, text_is_token=False, apply_pamr=False, background_func="weighted_average_sigmoid", lambda_bg=0.2,
    ):
        H, W = image.shape[2:]
        pH, pW = image.shape[2:]
        num_classes = text_emb.shape[0]
        batch_size = image.shape[0]

        image_rgb = image[:, [2, 1, 0], :, :]
        ori_image = image_rgb.clone()
        
        img_preprocessed = self.image_transforms(image_rgb).to(next(self.parameters()).device)
        
        # [修改] 使用 encode_image 统一处理特征提取 (包含多尺度逻辑)
        # 这样无论是单尺度还是多尺度，这里拿到的 image_feat 都是经过投影前的 Raw Feature (1024 or 3072)
        # 等等，encode_image 返回的是已经聚合过的 (B, Dim) 向量，丢失了空间信息
        # 分割任务需要 Patch Tokens！
        # 所以这里不能直接调 encode_image，必须手动重写一份支持 Spatial 的逻辑
        
        # === 1. 提取 Patch Features (保留空间维度) ===
        # 运行模型
        if self.multiscale:
            self.layer_hook_helper.clear()
        
        x_dict = self.model.forward_features(img_preprocessed)
        
        # 获取基础 Patch Tokens
        if isinstance(x_dict, dict):
            patch_tokens_last = x_dict['x_norm_patchtokens']
        else:
            patch_tokens_last = x_dict[:, self.num_global_tokens:, :]
            
        batch_size, num_tokens, last_embed_dim = patch_tokens_last.shape
        
        # 获取 Attention Maps
        self_attn, self_attn_maps = self.process_self_attention(
            self.feats['self_attn'], batch_size, num_tokens + self.num_global_tokens, 
            self.num_attn_heads, last_embed_dim, self.scale, self.num_global_tokens, ret_self_attn_maps=True
        )

        if self.multiscale:
            # === 多尺度空间特征 ===
            multi_scale_patches = []
            for layer_idx in self.target_layers:
                layer_out = self.layer_hook_helper.feats[layer_idx] # (B, Total, 1024)
                layer_patches = layer_out[:, self.num_global_tokens:, :]
                multi_scale_patches.append(layer_patches)
            
            # 拼接: (B, Num_Tokens, 3072)
            image_feat = torch.cat(multi_scale_patches, dim=-1)
        else:
            # === 单尺度空间特征 ===
            image_feat = patch_tokens_last # (B, Num_Tokens, 1024)

        # === 2. 投影层 ===
        if type(self.proj) == VisualProjectionLayer:
            image_feat = self.proj.project_dino(image_feat.float())
        if type(self.proj) == DoubleMLP:
            image_feat = self.proj.project_visual(image_feat.float())
            
        # Reshape 回空间尺寸
        b, np, c = image_feat.shape
        np_h = np_w = int(sqrt(np))
        image_feat = image_feat.reshape(b, np_h, np_w, c).permute(0, 3, 1, 2) # (B, C, H, W)
        
        # === 3. Mask 生成 ===
        mask, simmap = self.masker.forward_seg(image_feat, text_emb, hard=False)
        
        if self.with_bg_clean:
            mask = self.similarity_assignment_weighted(mask, image_feat, self_attn_maps, text_emb, lambda_bg)

        mask = F.interpolate(mask, (pH, pW), mode='bilinear', align_corners=True)

        if apply_pamr:
            for c in range(0, mask.shape[1], 30):
                mask[:, c:c + 30] = self.apply_pamr(ori_image, mask[:, c:c + 30])

        return mask, simmap

    # ... (其余辅助函数保持不变：build_dataset_class_tokens, build_text_embedding, apply_pamr, similarity_assignment_weighted)
    @torch.no_grad()
    def build_dataset_class_tokens(self, template_set, classnames):
        tokens = []
        templates = get_template(template_set)
        for classname in classnames:
            if 'bert' not in self.clip_model_name:
                tokens.append(
                    clip.tokenize([template.format(classname) for template in templates])
                )
            else:
                tokens.append(self.tokenizer([template.format(classname) for template in templates], return_tensors='pt', padding='max_length')['input_ids'])
        # [N, T, L], N: number of instance, T: number of captions (including ensembled), L: sequence length
        tokens = torch.stack(tokens)

        return tokens

    @torch.no_grad()
    def build_text_embedding(self, text):
        """
        Args:
            text (torch.Tensor): [NUM_CLASSES, NUM_TEMPLATES, CONTEXT_LENGTH] text tokens

        Returns:
            text_embs
        """
        text = text.to(next(self.parameters()).device)
        num_classes, num_templates = text.shape[:2]
        text_argmax = text.argmax(dim=-1)
        text_argmax = rearrange(text_argmax, 'n t -> (n t)', n=num_classes, t=num_templates)
        text = rearrange(text, 'n t l -> (n t) l', n=num_classes, t=num_templates)
        # chunked inference for memory limitation
        chunk_size = 32
        N = text.size(0)
        if type(self.proj) == CLIPLastLayer:
            text_embs = torch.cat([
            self.proj.project_clip_txt(self.encode_text(text[i:i + chunk_size]).permute(1, 0, 2), text_argmax=text_argmax[i:i + chunk_size])
            for i in range(0, N, chunk_size)
        ])
        else:
            if not self.use_avg_text_token:
                # performing classification using CLS textual token
                if 'bert' not in self.clip_model_name:
                    text_embs = torch.cat([
                        self.clip_model.encode_text(text[i:i + chunk_size])
                        for i in range(0, N, chunk_size)
                    ])
                else:
                    # encoding with BERT
                    text_embs = []
                    for i in range(0, N, chunk_size):
                        outputs = self.clip_model(text[i:i + chunk_size])
                        text_embs.append(outputs['pooler_output'])
                    text_embs = torch.cat(text_embs)
            else:
                # using text token average
                text_embs = []
                for i in range(0, N, chunk_size):
                    self.clip_model.encode_text(text[i:i + chunk_size])
                    text_embs.append(average_text_tokens(self.feats['clip_txt_out_tokens'] @ self.clip_model.text_projection, text[i:i + chunk_size] > 0, self.keep_cls, self.keep_end_seq))
                text_embs = torch.cat(text_embs)
        # [N, T, C]
        text_embs = rearrange(text_embs, '(n t) c -> n t c', n=num_classes, t=num_templates)
        # [N, C]
        text_embs = text_embs.mean(dim=1).float()
        if type(self.proj) == ProjectionLayer or type(self.proj) == DoubleMLP:
            text_embs = self.proj.project_clip_txt(text_embs)
        text_embs = us.normalize(text_embs, dim=-1)

        return text_embs

    def apply_pamr(self, image, mask):
        image = F.interpolate(image, mask.shape[-2:], mode="bilinear", align_corners=True)
        if self.pamr is None:
            pamr_iter = 10
            pamr_kernel = [1, 2, 4, 8, 12, 24]
            self.pamr = PAMR(pamr_iter, pamr_kernel)
            self.pamr.eval()
            self.pamr.to(next(self.parameters()).device)

        mask = self.pamr(image, mask)
        return mask

    def compute_padsize(self, H: int, W: int, patch_size: int):
        l, r, t, b = 0, 0, 0, 0
        if W % patch_size:
            lr = patch_size - (W % patch_size)
            l = lr // 2
            r = lr - l

        if H % patch_size:
            tb = patch_size - (H % patch_size)
            t = tb // 2
            b = tb - t

        return l, r, t, b

    def similarity_assignment_weighted(self, mask, image_feat, self_attn_maps, text_emb, lambda_bg=0.2):
        bs, c, h, w = image_feat.shape
        bs, num_classes, h, w = mask.shape
        bs, num_heads, hw = self_attn_maps.shape
        image_feat = image_feat.reshape(bs, c, hw)
        num_classes, c = text_emb.shape
        avg_head_embed = (self_attn_maps.unsqueeze(2) * image_feat.unsqueeze(1)).mean(dim=-1)
        avg_head_embed = avg_head_embed / avg_head_embed.norm(dim=-1, keepdim=True)
        avg_head_embed = avg_head_embed.permute(0, 2, 1) # [B, C, M]
        head_text_sim = text_emb.unsqueeze(0) @ avg_head_embed # [B, M, N]
        head_text_sim = (head_text_sim).softmax(dim=-1)
        head_text_sim_sum = head_text_sim.sum(dim=-1)
        
        self_attn_maps_repeat = self_attn_maps.unsqueeze(1).repeat(1, num_classes, 1, 1)
        head_text_sim_repeat = head_text_sim.unsqueeze(-1).repeat(1, 1, 1, hw)
        avg_self_attn_per_class = (self_attn_maps_repeat * head_text_sim_repeat).sum(dim=2) / head_text_sim_sum.unsqueeze(-1).repeat(1, 1, hw)
        avg_self_attn_per_class = avg_self_attn_per_class.softmax(dim=-1)
        
        min_self_attn = avg_self_attn_per_class.min().item()
        max_self_attn = avg_self_attn_per_class.max().item()
        max_self_attn = max(max_self_attn, max_self_attn - min_self_attn)
        avg_self_attn_per_class = avg_self_attn_per_class - min_self_attn
        avg_self_attn_per_class = avg_self_attn_per_class / max_self_attn
        avg_self_attn_per_class = avg_self_attn_per_class * (mask.max() - mask.min()) + mask.min()
        mask = mask.reshape(num_classes, hw) # [N, P]
        mask_output = (mask + lambda_bg * avg_self_attn_per_class).reshape(bs, num_classes, h, w) / (1 + lambda_bg)
        return mask_output