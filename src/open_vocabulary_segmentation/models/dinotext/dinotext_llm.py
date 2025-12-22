import itertools
import os
import pickle
from math import sqrt
import re
import yaml
import json
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
# [在 dinotext.py 开头添加]
import torchvision.transforms.functional as TF  #用于图像旋转

from models.builder import MODELS
from models.dinotext.pamr import PAMR
from models.dinotext.masker import DINOTextMasker
import us
from datasets import get_template

from src.model import ProjectionLayer, VisualProjectionLayer, CLIPLastLayer, DoubleMLP
from src.loss import Contrastive
from src.hooks import average_text_tokens, get_vit_out, feats

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@MODELS.register_module()
class DINOTextLLM(nn.Module):
    
    def get_self_attention(self, module, input, output):
        self.feats['self_attn'] = output
        
    def get_clip_second_last_dense_out(self, model: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
        self.feats['clip_second_last_out'] = output
        self.feats['clip_second_last_out'].to(dtype=torch.float32)
    
    def get_all_out_tokens(self, model: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
        self.feats['clip_txt_out_tokens'] = output
        
    def __init__(
            self, model_name, resize_dim, clip_model_name, proj_class, proj_name, proj_model, avg_self_attn_token=False, disentangled_self_attn_token=True, loss=None, pre_trained=True,
            unfreeze_last_text_layer=False, unfreeze_last_image_layer=False, is_eval=True, use_avg_text_token=False, keep_cls=False, keep_end_seq=False, with_bg_clean=False, **kwargs
    ):
        super().__init__()
        self.feats = {}
        self.model_name = model_name
        # loading the model
        
        if 'dinov2' in model_name:
            self.model_family = 'facebookresearch/dinov2' if 'dinov2' in model_name else 'facebookresearch/dino:main'
            self.model = torch.hub.load(self.model_family, model_name)                
            
        elif 'mae' in model_name or 'sam' in model_name or 'clip' in model_name or 'dino' in model_name:
            self.model = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=0,  # remove classifier nn.Linear
                img_size=resize_dim
            )
            
            if 'sam' in model_name:
                self.model.blocks[-1].register_forward_hook(get_vit_out)
        else:
            raise Exception("Unknown ViT model")
        # self.model.eval()
        mean = (0.485, 0.456, 0.406) if not 'clip' in model_name else (0.4815, 0.4578, 0.4082)
        std = (0.229, 0.224, 0.225) if not 'clip' in model_name else (0.2686, 0.2613, 0.2758)
        self.image_transforms = T.Compose([
            T.Resize((resize_dim, resize_dim)),
            lambda x: T.ToTensor()(x) if not isinstance(x, torch.Tensor) else x / 255.0,  # ensure tensor
            T.Normalize(mean, std),
        ])
        
        self.model.to(device)
        self.model.requires_grad_(False)
        
        self.clip_model_name = clip_model_name
        if 'bert' in self.clip_model_name:
            self.clip_model = BertModel.from_pretrained(self.clip_model_name, output_hidden_states = False)
            # load the corresponding wordtokenizer
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

        with open(os.path.join('configs', f"{proj_class}.yaml"), 'r') as config_file:
            config = yaml.safe_load(config_file)['model']
            
        ProjClass = getattr(importlib.import_module('src.model'), proj_model)
        self.proj = ProjClass.from_config(config)
        if type(self.proj) == CLIPLastLayer:
            self.clip_model.transformer.resblocks[-2].register_forward_hook(self.get_clip_second_last_dense_out)
        
            
        if pre_trained:
            self.proj.load_state_dict(torch.load(os.path.join("weights", f"{proj_name}.pth"), 'cpu'))
        self.proj.to(device)
        
        self.masker = DINOTextMasker(similarity_type="cosine")
        self.masker = self.masker.eval()
        
        self.pamr = None
                
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
            # in this case we register a forward hook with the aim of getting all the tokens and not only the cls
            self.clip_model.ln_final.register_forward_hook(self.get_all_out_tokens)
            self.keep_cls = keep_cls
            self.keep_end_seq = keep_end_seq        
            
        self.with_bg_clean = with_bg_clean    

    
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
        x = self.model(images, is_training=(self.avg_self_attn_token or self.disentangled_self_attn_token))
        batch_size, num_tokens, embed_dim = x['x_norm_patchtokens'].shape
        num_tokens = num_tokens + self.num_global_tokens
        if self.avg_self_attn_token or self.disentangled_self_attn_token:
            self_attn, self_attn_maps = self.process_self_attention(self.feats['self_attn'], batch_size, num_tokens, self.num_attn_heads, embed_dim, self.scale, self.num_global_tokens, ret_self_attn_maps=True)
        if self.avg_self_attn_token:
            x = (self_attn.unsqueeze(-1) * x['x_norm_patchtokens']).mean(dim=1)
        elif self.disentangled_self_attn_token:
            self_attn_maps = self_attn_maps.softmax(dim=-1)
            x = (x['x_norm_patchtokens'].unsqueeze(1) * self_attn_maps.unsqueeze(-1)).mean(dim=2)

        return x, self_attn_maps

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
        if cosine:
            img_embed = F.normalize(img_embed, p=2, dim=1)
            txt_embed = F.normalize(txt_embed, p=2, dim=1)
        sim = img_embed @ txt_embed.transpose(1, 0)
        if not ret_similarity_matrix:
            sim = sim[torch.eye(len(sim)) > 0.5] # only diagonal elements
        
        ret['contrastive_loss'] = self.contrastive_loss.compute_contrastive_loss(sim)
        
        return ret


    # @torch.no_grad()
    # def build_dataset_class_tokens(self, template_set, classnames):
    #     tokens = []
    #     templates = get_template(template_set)
    #     for classname in classnames:
    #         if 'bert' not in self.clip_model_name:
    #             tokens.append(
    #                 clip.tokenize([template.format(classname) for template in templates])
    #             )
    #         else:
    #             tokens.append(self.tokenizer([template.format(classname) for template in templates], return_tensors='pt', padding='max_length')['input_ids'])
    #     # [N, T, L], N: number of instance, T: number of captions (including ensembled), L: sequence length
    #     tokens = torch.stack(tokens)

    #     return tokens
    
    
    @torch.no_grad()
    def build_dataset_class_tokens(self, template_set, classnames, use_llm=False, desc_path=None):
        """
        构建类别 Token。支持 LLM 扩展描述。
        """
@torch.no_grad()
    def build_dataset_class_tokens(self, template_set, classnames, use_llm=False, desc_path=None):
        """
        构建类别 Token。支持 LLM 扩展描述。
        """
        tokens = []
        
        # [修复] 这里的 template_set 可能会被 main.py 传错，我们在这里做一个防御性编程
        # 如果传入的是 "ViT-B/16" 这种奇怪的东西，强制回退到 "imagenet"
        try:
            templates = get_template(template_set)
        except ValueError:
            print(f"[Warning] Template set '{template_set}' not found. Fallback to 'imagenet'.")
            templates = get_template("imagenet")
        
        # 加载 LLM 描述
        descriptions_dict = {}
        if use_llm and desc_path is not None:
            if os.path.exists(desc_path):
                print(f"[DINOTextLLM] Loading LLM descriptions from: {desc_path}")
                with open(desc_path, 'r') as f:
                    data = json.load(f)
                    # 尝试自动获取内层字典
                    if len(data) > 0:
                        descriptions_dict = list(data.values())[0]
            else:
                print(f"[Warning] Description path {desc_path} not found! Using raw class names.")

        for classname in classnames:
            # 1. 确定文本列表
            if use_llm and classname in descriptions_dict:
                texts_to_prompt = descriptions_dict[classname]
            else:
                texts_to_prompt = [classname]
            
            # 2. 生成所有 Prompt (描述 x 模板)
            all_prompts = []
            for text in texts_to_prompt:
                all_prompts.extend([template.format(text) for template in templates])
            
            # 3. Tokenize
            if 'bert' not in self.clip_model_name:
                tokens.append(clip.tokenize(all_prompts))
            else:
                tokens.append(self.tokenizer(all_prompts, return_tensors='pt', padding='max_length')['input_ids'])
        
        # build_text_embedding 会自动处理 dim=1 的平均
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
    
    # @torch.no_grad()
    # def generate_masks(
    #         self, image, img_metas, text_emb, classnames, text_is_token=False, apply_pamr=False, background_func="weighted_average_sigmoid", lambda_bg=0.2,
    #         # kp_w=0.3,
    # ):
    #     """Generate masks for each text embeddings

    #     Args:
    #         image [B, 3, H, W]

    #     Returns:
    #         softmask [B, N, H, W]: softmasks for each text embeddings
    #     """

    #     H, W = image.shape[2:]  # original image shape

    #     # padded image size
    #     pH, pW = image.shape[2:]
    #     num_classes = text_emb.shape[0]
    #     batch_size = image.shape[0]

    #     image = image[:, [2, 1, 0], :, :]  # BGR to RGB
    #     ori_image = image.clone()
        
    #     img_preprocessed = self.image_transforms(image).to(next(self.parameters()).device)
    #     if 'dinov2' in self.model_name:
    #         image_feat = self.model.forward_features(img_preprocessed)['x_norm_patchtokens']
    #     elif 'dinov3' in self.model_name:
    #         image_feat = self.model.forward_features(img_preprocessed)[:, 5:, :]
    #     elif 'mae' in self.model_name or 'clip' in self.model_name or 'dino' in self.model_name:
    #         image_feat = self.model.forward_features(img_preprocessed)[:, 1:, :]
    #     elif 'sam' in self.model_name:
    #         self.model.forward_features(img_preprocessed)
    #         image_feat = feats['vit_out'].reshape(feats['vit_out'].shape[0], feats['vit_out'].shape[1]**2, feats['vit_out'].shape[-1]) # BS x N_PATCHES x EMBED_DIM
              
    #     batch_size, num_tokens, embed_dim = image_feat.shape
    #     if type(self.proj) == VisualProjectionLayer:
    #         image_feat = self.proj.project_dino(image_feat.float())
    #     if type(self.proj) == DoubleMLP:
    #         image_feat = self.proj.project_visual(image_feat.float())
    #     b, np, c = image_feat.shape
    #     np_h = np_w = int(sqrt(np))
    #     image_feat = image_feat.reshape(b, np_h, np_w, c).permute(0, 3, 1, 2)
        
    #     self_attn, self_attn_maps = self.process_self_attention(self.feats['self_attn'], batch_size, num_tokens + self.num_global_tokens, self.num_attn_heads, embed_dim, self.scale, self.num_global_tokens, ret_self_attn_maps=True)
    #     mask, simmap = self.masker.forward_seg(image_feat, text_emb, hard=False)  # [B, N, H', W']
        
    #     if self.with_bg_clean:
    #         mask = self.similarity_assignment_weighted(mask, image_feat, self_attn_maps, text_emb, lambda_bg)

    #     # resize
    #     mask = F.interpolate(mask, (pH, pW), mode='bilinear', align_corners=True)  # [B, N, H, W]

    #     if apply_pamr:
    #         for c in range(0, mask.shape[1], 30):
    #             mask[:, c:c + 30] = self.apply_pamr(ori_image, mask[:, c:c + 30])

    #     assert mask.shape[2] == H and mask.shape[3] == W, f"shape mismatch: ({H}, {W}) / {mask.shape}"

    #     return mask, simmap
    # =======================================================
    
    
    # [新增/修改] TTA 核心逻辑开始
    # =======================================================
    @torch.no_grad()
    def generate_masks(
            self, image, img_metas, text_emb, classnames, text_is_token=False, apply_pamr=False, background_func="weighted_average_sigmoid", lambda_bg=0.2,
            # kp_w=0.3,
    ):
        """
        带 TTA (旋转增强) 的掩码生成函数
        """
        # 定义旋转角度列表：0度, 90度, 180度, 270度
        # 如果你想关闭 TTA，只需改为 angles = [0]
        angles = [0, 90, 180, 270] 
        
        accum_mask = None
        accum_simmap = None
        
        # 1. 循环执行多角度推理
        for angle in angles:
            # (1) 旋转输入图片
            if angle == 0:
                img_input = image
            else:
                img_input = TF.rotate(image, angle)
            
            # (2) 执行单次推理 (调用下面的 _inference_single)
            mask_i, simmap_i = self._inference_single(
                img_input, img_metas, text_emb, classnames, text_is_token, apply_pamr, background_func, lambda_bg
            )
            
            # (3) 将输出结果反向旋转回 0 度
            if angle != 0:
                # 注意：rotate 只能处理 H,W 在最后两维的张量
                mask_i = TF.rotate(mask_i, -angle)
                simmap_i = TF.rotate(simmap_i, -angle)
            
            # (4) 累加结果
            if accum_mask is None:
                accum_mask = mask_i
                accum_simmap = simmap_i
            else:
                accum_mask += mask_i
                accum_simmap += simmap_i
        
        # 2. 取平均值
        final_mask = accum_mask / len(angles)
        final_simmap = accum_simmap / len(angles)
        
        return final_mask, final_simmap

        # """
        # [增强版] 8x TTA: 4 角度旋转 x 2 镜像翻转
        # """
        # # 1. 定义 TTA 策略
        # angles = [0, 90, 180, 270] 
        # flips = [False, True]  # False: 原图, True: 水平翻转
        
        # accum_mask = None
        # accum_simmap = None
        # count = 0
        
        # # 2. 双重循环：遍历所有翻转和旋转的组合
        # for flip in flips:
        #     for angle in angles:
        #         count += 1
                
        #         # --- (A) 前向变换 (Pre-process) ---
        #         # 必须严格遵守顺序：先翻转，再旋转
        #         img_input = image
        #         if flip:
        #             img_input = TF.hflip(img_input)
        #         if angle > 0:
        #             img_input = TF.rotate(img_input, angle)
                
        #         # --- (B) 推理 (Inference) ---
        #         # 调用你现有的单次推理逻辑
        #         mask_i, simmap_i = self._inference_single(
        #             img_input, img_metas, text_emb, classnames, text_is_token, apply_pamr, background_func, lambda_bg
        #         )
                
        #         # --- (C) 反向变换 (Post-process) ---
        #         # 必须严格遵守“反向”顺序：先反旋转，再反翻转
        #         if angle > 0:
        #             mask_i = TF.rotate(mask_i, -angle)
        #             simmap_i = TF.rotate(simmap_i, -angle)
        #         if flip:
        #             mask_i = TF.hflip(mask_i)
        #             simmap_i = TF.hflip(simmap_i)
                
        #         # --- (D) 累加 ---
        #         if accum_mask is None:
        #             accum_mask = mask_i
        #             accum_simmap = simmap_i
        #         else:
        #             accum_mask += mask_i
        #             accum_simmap += simmap_i
        
        # # 3. 取平均
        # final_mask = accum_mask / count
        # final_simmap = accum_simmap / count
        
        # return final_mask, final_simmap


    def _inference_single(
            self, image, img_metas, text_emb, classnames, text_is_token=False, apply_pamr=False, background_func="weighted_average_sigmoid", lambda_bg=0.2,
    ):
        """
        [原 generate_masks 的核心逻辑] 
        这里只负责处理单张图片，不做改动，只是从原函数里剪切过来
        """
        H, W = image.shape[2:]  # original image shape

        # padded image size
        pH, pW = image.shape[2:]
        num_classes = text_emb.shape[0]
        batch_size = image.shape[0]

        # 注意：这里可能需要确保 image 是 Tensor，原逻辑里有 transforms
        # 如果输入的已经是预处理过的 tensor，这里要注意 transform 是否会重复
        # 通常 MMSeg 传进来的是 Tensor (B, 3, H, W)
        
        # 原逻辑开始 --------------------------------------------
        # image = image[:, [2, 1, 0], :, :]  # BGR to RGB (MMSeg读取可能是BGR)
        # 注意：如果 image 已经是 Tensor，切片操作是通用的
        # 为了保险，保留原逻辑的 tensor 操作
        image_rgb = image[:, [2, 1, 0], :, :]
        ori_image = image_rgb.clone()
        
        # 预处理
        img_preprocessed = self.image_transforms(image_rgb).to(next(self.parameters()).device)
        
        # 提取特征
        if 'dinov2' in self.model_name:
            image_feat = self.model.forward_features(img_preprocessed)['x_norm_patchtokens']
        elif 'dinov3' in self.model_name:
            image_feat = self.model.forward_features(img_preprocessed)[:, 5:, :]
        elif 'mae' in self.model_name or 'clip' in self.model_name or 'dino' in self.model_name:
            image_feat = self.model.forward_features(img_preprocessed)[:, 1:, :]
        elif 'sam' in self.model_name:
            self.model.forward_features(img_preprocessed)
            image_feat = feats['vit_out'].reshape(feats['vit_out'].shape[0], feats['vit_out'].shape[1]**2, feats['vit_out'].shape[-1])
              
        batch_size, num_tokens, embed_dim = image_feat.shape
        
        # 投影层
        if type(self.proj) == VisualProjectionLayer:
            image_feat = self.proj.project_dino(image_feat.float())
        if type(self.proj) == DoubleMLP:
            image_feat = self.proj.project_visual(image_feat.float())
            
        b, np, c = image_feat.shape
        np_h = np_w = int(sqrt(np))
        image_feat = image_feat.reshape(b, np_h, np_w, c).permute(0, 3, 1, 2)
        
        # 注意力处理
        self_attn, self_attn_maps = self.process_self_attention(
            self.feats['self_attn'], batch_size, num_tokens + self.num_global_tokens, 
            self.num_attn_heads, embed_dim, self.scale, self.num_global_tokens, ret_self_attn_maps=True
        )
        
        # 生成 Mask
        mask, simmap = self.masker.forward_seg(image_feat, text_emb, hard=False)
        
        # 背景清洗 (Background Cleaning)
        if self.with_bg_clean:
            # 确保传入正确的 lambda_bg
            mask = self.similarity_assignment_weighted(mask, image_feat, self_attn_maps, text_emb, lambda_bg)

        # 上采样
        mask = F.interpolate(mask, (pH, pW), mode='bilinear', align_corners=True)

        if apply_pamr:
            for c in range(0, mask.shape[1], 30):
                mask[:, c:c + 30] = self.apply_pamr(ori_image, mask[:, c:c + 30])

        assert mask.shape[2] == H and mask.shape[3] == W, f"shape mismatch: ({H}, {W}) / {mask.shape}"
        
        return mask, simmap
    # =======================================================
    # [新增/修改] TTA 核心逻辑结束
    # =======================================================


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
