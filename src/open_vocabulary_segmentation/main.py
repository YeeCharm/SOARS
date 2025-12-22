# ------------------------------------------------------------------------------
# Talk2DINO
# ------------------------------------------------------------------------------
# Modified from GroupViT (https://github.com/NVlabs/GroupViT)
# Copyright (c) 2021-22, NVIDIA Corporation & affiliates. All Rights Reserved.
# ------------------------------------------------------------------------------
import sys
from unittest.mock import MagicMock

# Mock mmcv.ops to bypass missing CUDA extensions in pure Python environment
try:
    import mmcv.ops
except (ImportError, ModuleNotFoundError):
    # Create a mock module for mmcv.ops
    mock_ops = MagicMock()
    sys.modules["mmcv.ops"] = mock_ops
    # Ensure point_sample and other ops are available as attributes of the mock
    # This handles 'from mmcv.ops import point_sample'
    # Note: This allows the code to import, but execution of these ops will fail/do nothing.
    # For evaluation with DINOv2 + MLP, these ops might not be used.

import argparse
import datetime
import json
import os
import os.path as osp
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
from torch.distributed.distributed_c10d import _get_default_group
import numpy as np
# from mmcv.parallel import MMDistributedDataParallel
# from mmcv.runner import get_dist_info, init_dist, set_random_seed
# from mmcv.utils import collect_env, get_git_hash
from mmengine.model import MMDistributedDataParallel
from mmengine.dist import get_dist_info, init_dist
from mmengine.runner import set_random_seed
from mmengine.utils import get_git_hash
from mmengine.utils.dl_utils import collect_env
# from mmseg.apis import multi_gpu_test
from torch.utils.data import Subset

# Initialize mmseg registry FIRST
try:
    from mmseg.utils import register_all_modules
    register_all_modules(init_default_scope=True)
except ImportError:
    print("Warning: Could not import register_all_modules from mmseg.utils")

# from datasets import build_loader, build_text_transform
from models import build_model
from omegaconf import OmegaConf, read_write

from segmentation.evaluation import build_seg_dataloader, build_seg_dataset, build_dinotext_seg_inference
import segmentation.evaluation.builder as builder_module

from timm.utils import AverageMeter
from torchvision.utils import make_grid
from utils import (
    build_optimizer,
    build_scheduler,
    get_config,
    get_grad_norm,
    get_logger,
    parse_losses,
    load_config
)
import us
from utils import (
    build_optimizer,
    build_scheduler,
    get_config,
    get_grad_norm,
    get_logger,
    load_checkpoint,
    parse_losses,
    save_checkpoint,
    CheckpointManager,
    load_config
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# from mmseg.datasets import PIPELINES, PascalVOCDataset, PascalContextDataset, ADE20KDataset, CityscapesDataset, \
#     COCOStuffDataset, PascalContextDataset59
# Note: PascalContextDataset and PascalContextDataset59 might need to be imported differently or are not available in standard mmseg 1.x
# If they are custom datasets, they should be registered.
# For now, we import what is available.

# @PIPELINES.register_module()
# class FloatImage:
#     def __call__(self, results):
#         results['img'] = results['img'].astype(np.float32)
#         return results




def cyclize(loader):
    while True:
        for i in loader:
            yield i


def get_argparser():
    parser = argparse.ArgumentParser("DINOText training and evaluation script")
    parser.add_argument("--cfg", type=str, help="path to config file")
    parser.add_argument(
        "--opts", help="Modify config options by adding 'KEY=VALUE' list. ", default=None, nargs="+"
    )

    # [新增] LLM 扩展相关参数
    parser.add_argument("--use_llm", action="store_true", help="Enable LLM-based text description expansion")
    parser.add_argument("--desc_path", type=str, default=None, help="Path to the class description JSON file")

    # easy config modification
    parser.add_argument("--batch-size", type=int, help="batch size for single GPU")
    parser.add_argument(
        "--output",
        type=str,
        help="root of output folder, " "the full path is <output>/<model_name>/<tag>",
    )
    parser.add_argument("--tag", type=str, help="tag of experiment")
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument("--wandb", action="store_true", help="Use W&B to log experiments")
    parser.add_argument("--wandb_name", type=str, help="W&B run name", default="default")
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--eval_cfg', type=str, default="configs/dinotext.yml")
    parser.add_argument("--eval_base_cfg", type=str, default="configs/eval.yml")

    parser.add_argument("--pred_qual_path", type=str, default=None)
    parser.add_argument("--gt_qual_path", type=str, default=None)

    parser.add_argument("--job_id", type=int, default=0)
    parser.add_argument("--num_jobs", type=int, default=1)

    return parser

def log_results(miou, proj_name, bench, result_dir, logger):
    os.makedirs(result_dir, exist_ok=True)
    
    json_path = os.path.join(result_dir, f"{proj_name}.json")
    
    # Load existing data or start with an empty dictionary
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                content = f.read()
                if content.strip():
                    data = json.loads(content)
                else:
                    data = {}
        except json.JSONDecodeError:
            logger.warning(f"Could not decode JSON from {json_path}. Starting with empty data.")
            data = {}
    else:
        data = {}
    
    # Add or update the benchmark result
    # Ensure miou is a standard float, not numpy float
    if hasattr(miou, 'item'):
        miou = miou.item()
    elif isinstance(miou, (np.float32, np.float64)):
        miou = float(miou)
        
    data[bench] = miou
    
    # Write the updated data to the JSON file in human-readable format
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved results at {json_path}")

def train(cfg, args):
    if device == "cuda":
        dist.barrier()

    # build datasets
    # dataset_train, data_loader_train = build_loader(cfg.data) # TODO: Ripristinate something like this
    # ___________________________________________
    # TODO
    # ___________________________________________
    from torch.utils.data import DataLoader
    import torchvision.transforms as T
    import clip
    import sys
    sys.path.append("src")
    from src.dataset import COCOCaptions

    image_transforms = T.Compose([
        T.Resize(448, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(448),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # dataset_train = COCOCaptions('coco/train.json', 'coco/train2014', "train", image_transforms, clip.tokenize)
    # data_loader_train = DataLoader(dataset_train, batch_size=cfg.data.batch_size, shuffle=True)
    data_loader_train = None
    # ___________________________________________
    # End TODO
    # ___________________________________________

    # build validation loaders
    val_loaders = {}
    for key in cfg.evaluate.task:
        if key == "cls":
            continue

        dataset = build_seg_dataset(cfg.evaluate.get(key))
        len_dataset = len(dataset)

        first_sample = args.job_id * len_dataset // args.num_jobs
        last_sample = ((args.job_id + 1) * len_dataset // args.num_jobs)
        if args.job_id == args.num_jobs - 1:
            last_sample = len_dataset

        dataset = Subset(dataset, range(first_sample, last_sample))
        loader = build_seg_dataloader(dataset)
        val_loaders[key] = loader

    logger = get_logger()

    # build model & optimizer
    logger.info(f"Creating model:{cfg.model.type}/{cfg.model_name}")
    model = build_model(cfg.model)
    if device == "cuda":
        model.cuda()

        # model.set_train(decoder_only=(cfg.train.ust_steps > 0), config=cfg)
        # optimizer = build_optimizer(cfg.train, model)
        import torch.optim as optim
        optimizer = optim.Adam(model.parameters(), lr=cfg.train.base_lr) # TODO: Ripristinate
        model = MMDistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters} ({n_parameters/1000/1000:.1f}M)")
    lr_scheduler = build_scheduler(cfg.train, optimizer)

    # fp16 compression
    logger.info(us.dist_info())
    if cfg.train.fp16 and cfg.train.fp16_comm:
        if int(os.getenv("LOCAL_WORLD_SIZE")) < int(os.getenv("WORLD_SIZE")):
            pg = _get_default_group()
            logger.info("!!! Multi-node setting :: turn on fp16 compression hook")
            model.register_comm_hook(pg, fp16_compress_hook)
        else:
            logger.info("!!! Single-node setting :: skip fp16 compression hook")

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.fp16)

    if cfg.checkpoint.resume:
        load_checkpoint(cfg, model.module, optimizer, lr_scheduler, scaler)


    if cfg.evaluate.eval_only:
        # res = evaluate(cfg, model, val_loaders)
        
        res = evaluate(cfg, model, val_loaders, args)
        logger.info(res)
        # if "metrics" in res, assign to metrics and remove it
        metrics = res.pop("metrics", None)
        r = ", ".join([f"{v:.2f}" for v in res.values()])
        logger.info(f" >> {r}")
        logger.info(f"Experiment dir: {cfg.output}")
        # log res on wandb as statics
        if cfg.wandb and metrics:
            import wandb
            wandb.init(
                project="open-vocab-metrics",
                name=args.wandb_name,
                dir=cfg.output,
                config=OmegaConf.to_container(cfg, resolve=True),
                resume=False,
            )
            wandb.log(metrics[0])
        log_results(res['val/avg_miou'], cfg['model']['proj_name'], cfg['evaluate']['task'][0], 'segmentation_results', logger)
        return

    logger.info("Start training")
    start_time = time.time()

    do_training(cfg, model, data_loader_train, optimizer, lr_scheduler, scaler, val_loaders)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))
    if device == "cuda":
        dist.barrier()


def do_training(config, model, data_loader, optimizer, lr_scheduler, scaler, val_loaders):
    logger = get_logger()
    dist.barrier()
    model.train()
    optimizer.zero_grad()
    if config.wandb and dist.get_rank() == 0:
        import wandb
    else:
        wandb = None

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    log_vars_meters = defaultdict(AverageMeter)

    total_steps = config.train.total_steps
    org_total_steps = total_steps
    # update training steps by evaluation step (discard non-evaluation steps)
    total_steps = total_steps - (total_steps % config.evaluate.eval_freq) + 1
    if org_total_steps != total_steps:
        logger.info(f"Total step is updated: {org_total_steps} -> {total_steps}")
        
    ckpt_manager = CheckpointManager(config.checkpoint.save_topk, config.output)

    batch_size = config.data.batch_size
    accum_freq = config.train.accum_freq

    if accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    # ust_check = True
    end = time.time()
    for step, samples in enumerate(cyclize(data_loader), config.train.start_step):
        if step >= total_steps:
            break
        # if ust_check and config.train.ust_steps and step >= config.train.ust_steps:
        #     model.module.set_train(decoder_only=False, config=config)
        #     logger.info(f" -- [{step}] UST stage is DONE; Now fine-tuning stage begins ...")
        #     ust_check = False

        # caption = samples.pop("org_caption")

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=config.train.fp16):
            # losses = model(**samples)
            img_emb, txt_emb = model(image=samples["image"], text=samples["annotation"])
            losses = model.compute_loss(img_emb, txt_emb)

        loss, log_vars = parse_losses(losses)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # if config.train.clip_grad:
        #     grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad)
        # else:
        #     grad_norm = get_grad_norm(model.parameters())

        scaler.step(optimizer)
        scaler.update()

        lr_scheduler.step()
        torch.cuda.synchronize()

        loss_meter.update(loss.item(), batch_size)
        for loss_name in log_vars:
            log_vars_meters[loss_name].update(log_vars[loss_name].item(), batch_size)
        # norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if step % config.print_freq == 0:
            lr = optimizer.param_groups[0]["lr"]
            epoch = step / num_steps
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            #  etas = batch_time.avg * (num_steps - step)
            log_vars_str = "  ".join(
                f"{n} {m.val:7.6f} ({m.avg:7.6f})" for n, m in log_vars_meters.items()
            )
            logger.info(
                f"Train: [EP {epoch:.1f}][{step:6d}/{total_steps}]  "
                #  f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f"time {batch_time.val:.3f} ({batch_time.avg:.3f})  "
                f"total_loss {loss_meter.val:7.6f} ({loss_meter.avg:7.6f})  "
                f"{log_vars_str}  "
                # f"grad_norm {norm_meter.val:7.4f} ({norm_meter.avg:7.4f})  "
                f"lr {lr:.6f}  "
                f"mem {memory_used:.0f}MB"
            )

            if wandb is not None:
                log_stat = {f"iter/train_{n}": m.val for n, m in log_vars_meters.items()}
                log_stat["iter/train_total_loss"] = loss_meter.val
                # log_stat["iter/grad_norm"] = norm_meter.val
                log_stat["iter/learning_rate"] = lr
                log_stat["iter/epoch"] = epoch
                log_stat["iter/grad_scale"] = scaler.get_scale()

                # image & mask logging
                # if "mask" in losses and step % 500 == 0:
                #     N = 3

                #     # un-normalize image
                #     org_img = us.unnorm(samples["image"][:N])
                #     org_img = torch.clamp(org_img, 0.0, 1.0)  # random erasing makes out-of-range value
                #     mask = losses["mask"][:N].repeat(1, 3, 1, 1).cpu().float()
                #     mask = F.interpolate(mask, org_img.shape[2:]) > 0.5
                #     log_images = [org_img, mask, org_img * mask]
                #     if "neg_mask" in losses:
                #         neg_mask = losses["neg_mask"][:N, :1].repeat(1, 3, 1, 1).cpu().float()
                #         neg_mask = F.interpolate(neg_mask, org_img.shape[2:]) > 0.5
                #         log_images.append(neg_mask)

                #     log_images = torch.cat(log_images)
                #     grid = make_grid(log_images, nrow=N, value_range=(0, 1))
                #     cap = "\n".join([f"[{i}] {c}" for i, c in enumerate(caption[:N])])
                #     log_stat["examples"] = wandb.Image(grid, caption=cap)

                wandb.log(log_stat, step=step)

        if step and step % config.evaluate.eval_freq == 0:
            metrics = evaluate(config, model, val_loaders)

            if us.is_global_zero():
                ckpt_kwargs = {
                    "config": config,
                    "step": step,
                    "model": model,
                    "optimizer": optimizer,
                    "lr_scheduler": lr_scheduler,
                    "scaler": scaler,
                    "metrics": metrics,
                }
                save_checkpoint(**ckpt_kwargs)
                if config.checkpoint.save_all:
                    save_checkpoint(**ckpt_kwargs, filename=f"ckpt_{step}.pth")
                # save best
                if config.checkpoint.save_topk:
                    miou = metrics["val/avg_miou"]
                    ckpt_manager.add(miou, ckpt_kwargs, step)

            dist.barrier()

            if wandb is not None:
                wandb.log(metrics, step=step)

            batch_time.reset()
            loss_meter.reset()
            norm_meter.reset()
            for m in log_vars_meters.values():
                m.reset()


@torch.no_grad()
def evaluate(cfg, model, val_loaders, args=None):
    logger = get_logger()
    ret = {}
    model.eval()

    for key, loader in val_loaders.items():
        if key == "cls":
            continue

        dataset_class = loader.dataset.__class__.__name__
        logger.info(f"### Validation dataset: {key} ({dataset_class})")

        # miou, metrics = validate_seg(cfg, cfg.evaluate.get(key), loader, model)
        
        miou, metrics = validate_seg(cfg, cfg.evaluate.get(key), loader, model, args)

        logger.info(f"[{key}] mIoU of {len(loader.dataset)} test images: {miou:.2f}%")
        ret[f"val/{key}_miou"] = miou
        ret[f"metrics"] = metrics

    ret["val/avg_miou"] = np.mean([v for k, v in ret.items() if "miou" in k])

    model.train()

    return ret


# @torch.no_grad()
# def validate_seg(config, seg_config, data_loader, model):
#     logger = get_logger()
#     if device == "cuda":
#         dist.barrier()

#     model.eval()

#     if hasattr(model, "module"):
#         model_without_ddp = model.module
#     else:
#         model_without_ddp = model

#     seg_model = build_dinotext_seg_inference(
#         model_without_ddp,
#         data_loader.dataset,
#         config,
#         seg_config,
#     )

#     if device == "cuda":
#         mmddp_model = MMDistributedDataParallel(
#             seg_model, device_ids=[torch.cuda.current_device()], broadcast_buffers=False
#         )
#     else:
#         mmddp_model = seg_model
#     mmddp_model.eval()

#     # TODO: Use multi-gpu-test from mmseg instead of ours
#     results, pred_qualitatives, gt_qualitatives, num_classes = us.multi_gpu_test(
#         model=mmddp_model,
#         data_loader=data_loader,
#         tmpdir=None,
#         gpu_collect=False, # device == "cuda",
#         efficient_test=False,
#         pre_eval=True,
#         format_only=False,
#     )

#     if device == "cpu" or dist.get_rank() == 0:
#         from segmentation.evaluation.custom_eval import evaluate_dataset
#         metric = [evaluate_dataset(data_loader.dataset, results, logger=logger)]
#     else:
#         metric = [None]

#     if device == "cuda":
#         dist.broadcast_object_list(metric)
#     miou_result = metric[0]["mIoU"] * 100

#     torch.cuda.empty_cache()
#     if device == "cuda":
#         dist.barrier()
#     return miou_result, metric



@torch.no_grad()
def validate_seg(config, seg_config, data_loader, model, args=None):
    logger = get_logger()
    if device == "cuda":
        dist.barrier()

    model.eval()



    if hasattr(model, "module"):
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    # -------------------------------------------------------------
    # [关键修复 1] 稳健地获取数据集 CLASSES (解决 Subset 报错)
    # -------------------------------------------------------------
    real_dataset = data_loader.dataset
    while isinstance(real_dataset, Subset):
        real_dataset = real_dataset.dataset
    
    CLASSES = None
    # 依次尝试获取 CLASSES
    if hasattr(real_dataset, 'CLASSES'):
        CLASSES = real_dataset.CLASSES
    elif hasattr(real_dataset, 'metainfo') and 'classes' in real_dataset.metainfo:
        CLASSES = real_dataset.metainfo['classes']
    elif hasattr(real_dataset, 'METAINFO') and 'classes' in real_dataset.METAINFO:
        CLASSES = real_dataset.METAINFO['classes']
    else:
        try: CLASSES = real_dataset.__class__.CLASSES
        except: pass
            
    if CLASSES is None:
        # 如果实在找不到，打印个警告，但这通常不会发生
        logger.warning(f"Could not find 'CLASSES' in dataset {type(real_dataset)}. LLM expansion might fail.")
        CLASSES = [] 
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # [关键修复 2] 注入 LLM 逻辑并手动生成 Embedding
    # -------------------------------------------------------------
    use_llm = False
    desc_path = None
    if args is not None:
        use_llm = getattr(args, "use_llm", False)
        desc_path = getattr(args, "desc_path", None)

    # 这里的第一个参数必须是 'imagenet'，不能是 config.model.clip_model_name
    template_key = 'full'

    text_embeddings = None
    if use_llm and desc_path:
        if dist.get_rank() == 0:
            logger.info(f"*** Pre-computing LLM Features... ***")
        
        # 这一步只做预计算，不影响主模型的 text_embeddings
        model_without_ddp.prepare_llm_features(desc_path, CLASSES, template_key)

        # 手动调用修改后的 build_dataset_class_tokens
        class_tokens = model_without_ddp.build_dataset_class_tokens(
            template_key, 
            CLASSES
        )
        text_embeddings = model_without_ddp.build_text_embedding(class_tokens)
    # -------------------------------------------------------------

    # 构建分割推理模型
    seg_model = build_dinotext_seg_inference(
        model_without_ddp,
        data_loader.dataset, # 传 Subset 没问题，内部只用来取长度等
        config,
        seg_config,
    )

    # [关键步骤] 强制覆盖 seg_model 里的 text_embedding
    if text_embeddings is not None:
        if hasattr(seg_model, 'text_embedding'):
            seg_model.text_embedding = text_embeddings
        elif hasattr(seg_model, 'module') and hasattr(seg_model.module, 'text_embedding'):
            seg_model.module.text_embedding = text_embeddings
        elif hasattr(seg_model, 'decode_head') and hasattr(seg_model.decode_head, 'text_embedding'):
             # 有些架构 embedding 存在 head 里
            seg_model.decode_head.text_embedding = text_embeddings

    if device == "cuda":
        mmddp_model = MMDistributedDataParallel(
            seg_model, device_ids=[torch.cuda.current_device()], broadcast_buffers=False
        )
    else:
        mmddp_model = seg_model
    mmddp_model.eval()

    results, pred_qualitatives, gt_qualitatives, num_classes = us.multi_gpu_test(
        model=mmddp_model,
        data_loader=data_loader,
        tmpdir=None,
        gpu_collect=False,
        efficient_test=False,
        pre_eval=True,
        format_only=False,
    )

    if device == "cpu" or dist.get_rank() == 0:
        from segmentation.evaluation.custom_eval import evaluate_dataset
        # 这里必须用 real_dataset，因为它包含用于评估的标签映射
        metric = [evaluate_dataset(real_dataset, results, logger=logger)]
    else:
        metric = [None]

    if device == "cuda":
        dist.broadcast_object_list(metric)
    miou_result = metric[0]["mIoU"] * 100

    torch.cuda.empty_cache()
    if device == "cuda":
        dist.barrier()
    return miou_result, metric


def main():
    parser = get_argparser()
    args = parser.parse_args()

    if args.eval:
        # update config when resume
        # default config -> org config -> eval config
        if args.cfg:
            default_cfg = load_config(args.cfg)
        else:
            default_cfg = load_config(args.eval_cfg)
        # default_cfg = load_config("configs/HOME.yml")
        # default_cfg = load_config("configs/freeda.yml")
        # org_cfg_path = Path(args.resume).parent / "config.json"
        # if org_cfg_path.exists():
        #     org_cfg = OmegaConf.load(Path(args.resume).parent / "config.json")
        # else:
        org_cfg = OmegaConf.create()  # empty container
        eval_cfg = OmegaConf.load(args.eval_base_cfg)
        cfg = OmegaConf.merge(default_cfg, org_cfg, eval_cfg)
        if args.opts is not None:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.opts))

        cfg.wandb = args.wandb
        cfg.evaluate.eval_only = args.eval
        args.output = args.output if args.output is not None else "output/eval"

        assert args.output is not None, "Please specify output folder for evaluation"
        cfg.output = args.output

        # create output folder if it does not exist
        Path(cfg.output).mkdir(parents=True, exist_ok=True)
    else:
        cfg = get_config(args)


    # TODO: The config can be modified here

    if device == "cuda":

        # start faster ref: https://github.com/open-mmlab/mmdetection/pull/7036
        mp.set_start_method("fork", force=True)
        init_dist("pytorch")
        rank, world_size = get_dist_info()
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")

        dist.barrier()

    else:
        rank = 0
        world_size = 1

    # set_random_seed(cfg.seed, use_rank_shift=True)
    set_random_seed(cfg.seed)
    cudnn.benchmark = True

    os.makedirs(cfg.output, exist_ok=True)
    logger = get_logger(cfg)

    # linear scale the learning rate according to total batch size, may not be optimal
    # linear_scaled_lr = cfg.train.base_lr * cfg.data.batch_size * world_size / 4096.0
    # linear_scaled_min_lr = cfg.train.min_lr * cfg.data.batch_size * world_size / 4096.0

    # with read_write(cfg):
    #     logger.info(f"Scale base_lr from {cfg.train.base_lr} to {linear_scaled_lr}")
    #     logger.info(f"Scale min_lr from {cfg.train.min_lr} to {linear_scaled_min_lr}")
    #     cfg.train.base_lr = linear_scaled_lr
    #     cfg.train.min_lr = linear_scaled_min_lr

    if device == "cuda" and dist.get_rank() == 0:
        path = os.path.join(cfg.output, "config.json")
        OmegaConf.save(cfg, path)
        logger.info(f"Full config saved to {path}")

    # log env info
    env_info_dict = collect_env()
    env_info = "\n".join([f"{k}: {v}" for k, v in env_info_dict.items()])
    dash_line = "-" * 60 + "\n"
    logger.info("Environment info:\n" + dash_line + env_info + "\n" + dash_line)

    logger.info(f"Git hash: {get_git_hash(digits=7)}")

    # print config
    logger.info(OmegaConf.to_yaml(cfg))

    train(cfg, args)
    if device == "cuda":
        dist.barrier()

    # print outputdir
    logger.info(f"Experiment dir: {cfg.output}")


if __name__ == "__main__":
    main()
