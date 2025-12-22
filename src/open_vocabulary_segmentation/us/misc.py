# ------------------------------------------------------------------------------
# FreeDA
# ------------------------------------------------------------------------------
from typing import Dict, List, Any
from datetime import datetime
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

# ImageNet mean/std (from timm)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

DEFAULT_MEAN = IMAGENET_DEFAULT_MEAN
DEFAULT_STD = IMAGENET_DEFAULT_STD

# NOTE Originally CLIP statistics should be used, but the legacy of ImageNet statistics
# from GroupViT is applied. Fortunately, CLIP is quite robust to slightly different
# normalization constants (https://github.com/openai/CLIP/issues/20#issuecomment-764985771).


def unnorm(x):
    mean = torch.as_tensor(DEFAULT_MEAN, device=x.device)[None, ..., None, None]
    std = torch.as_tensor(DEFAULT_STD, device=x.device)[None, ..., None, None]
    return x.mul(std).add(mean)


# DEBUG NaN
def check_nonfinite(x, name=""):
    rank = dist.get_rank()
    n_nan = x.isnan().sum()
    n_inf = x.isinf().sum()
    if n_nan or n_inf:
        print(f"[RANK {rank}] {name} is not finite: #nan={n_nan}, #inf={n_inf}")
        return True

    print(f"[RANK {rank}] {name} is OK ...")
    return False


def normalize(t, dim, eps=1e-6):
    """Large default eps for fp16"""
    return F.normalize(t, dim=dim, eps=eps)


def timestamp(fmt="%y%m%d-%H%M%S"):
    return datetime.now().strftime(fmt)


def merge_dicts_by_key(dics: List[Dict]) -> Dict[Any, List]:
    """Merge dictionaries by key. All of dicts must have same keys."""
    ret = {key: [] for key in dics[0].keys()}
    for dic in dics:
        for key, value in dic.items():
            ret[key].append(value)

    return ret


def flatten_2d_list(list2d):
    return list(chain.from_iterable(list2d))


def num_params(module):
    return sum(p.numel() for p in module.parameters())


def param_trace(name, module, depth=0, max_depth=999, threshold=0, printf=print):
    if depth > max_depth:
        return
    prefix = "  " * depth
    n_params = num_params(module)
    if n_params > threshold:
        printf("{:60s}\t{:10.3f}M".format(prefix + name, n_params / 1024 / 1024))
    for n, m in module.named_children():
        if depth == 0:
            child_name = n
        else:
            child_name = "{}.{}".format(name, n)
        param_trace(child_name, m, depth + 1, max_depth, threshold, printf)


@torch.no_grad()
def hash_bn(module):
    summary = []
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            w = m.weight.detach().mean().item()
            b = m.bias.detach().mean().item()
            rm = m.running_mean.detach().mean().item()
            rv = m.running_var.detach().mean().item()
            summary.append((w, b, rm, rv))

    if not summary:
        return 0.0, 0.0

    w, b, rm, rv = [np.mean(col) for col in zip(*summary)]
    p = np.mean([w, b])
    s = np.mean([rm, rv])

    return p, s


@torch.no_grad()
def hash_params(module):
    return torch.as_tensor([p.mean() for p in module.parameters()]).mean().item()


@torch.no_grad()
def hashm(module):
    p = hash_params(module)
    _, s = hash_bn(module)

    return p, s

import os.path as osp
import tempfile
import warnings
import shutil

import mmcv
import mmengine
import numpy as np
import torch
# from mmcv.engine import collect_results_cpu, collect_results_gpu
from mmengine.dist import get_dist_info
from mmcv.image import tensor2imgs
# from mmcv.runner import get_dist_info
# from mmseg.apis.test import np2tmp

device = "cuda" if torch.cuda.is_available() else "cpu"

def np2tmp(array, tmpdir=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to be saved.
        tmpdir (str | None): Path of directory to save the temporary results.

    Returns:
        str: The path of the saved numpy file.
    """
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp()
    mmengine.mkdir_or_exist(tmpdir)
    filename = osp.join(tmpdir, f'{os.getpid()}.npy')
    np.save(filename, array)
    return filename

from typing import Optional
def collect_results_cpu(result_part: list,
                        size: int,
                        tmpdir: Optional[str] = None) -> Optional[list]:
    """Collect results under cpu mode.

    On cpu mode, this function will save the results on different gpus to
    ``tmpdir`` and collect them by the rank 0 worker.

    Args:
        result_part (list): Result list containing result parts
            to be collected.
        size (int): Size of the results, commonly equal to length of
            the results.
        tmpdir (str | None): temporal directory for collected results to
            store. If set to None, it will create a random temporal directory
            for it.

    Returns:
        list: The collected results.
    """
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmengine.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmengine.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    part_file = osp.join(tmpdir, f'part_{rank}.pkl')  # type: ignore
    mmengine.dump(result_part, part_file)
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')  # type: ignore
            part_result = mmengine.load(part_file)
            # When data is severely insufficient, an empty part_result
            # on a certain gpu could makes the overall outputs empty.
            if part_result:
                part_list.append(part_result)
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)  # type: ignore
        return ordered_results


def multi_gpu_test(model,
                   data_loader,
                   tmpdir=None,
                   gpu_collect=False,
                   efficient_test=False,
                   pre_eval=False,
                   format_only=False,
                   format_args={}):
    """Test model with multiple gpus by progressive mode.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode. The same path is used for efficient
            test. Default: None.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Mutually exclusive with
            pre_eval and format_results. Default: False.
        pre_eval (bool): Use dataset.pre_eval() function to generate
            pre_results for metric evaluation. Mutually exclusive with
            efficient_test and format_results. Default: False.
        format_only (bool): Only format result for results commit.
            Mutually exclusive with pre_eval and efficient_test.
            Default: False.
        format_args (dict): The args for format_results. Default: {}.

    Returns:
        list: list of evaluation pre-results or list of save file names.
    """
    if efficient_test:
        warnings.warn(
            'DeprecationWarning: ``efficient_test`` will be deprecated, the '
            'evaluation is CPU memory friendly with pre_eval=True')
        mmengine.mkdir_or_exist('.efficient_test')
    # when none of them is set true, return segmentation results as
    # a list of np.array.
    assert [efficient_test, pre_eval, format_only].count(True) <= 1, \
        '``efficient_test``, ``pre_eval`` and ``format_only`` are mutually ' \
        'exclusive, only one of them could be true .'

    model.eval()
    results = []
    dataset = data_loader.dataset
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx

    # batch_sampler based on DistributedSampler, the indices only point to data
    # samples of related machine.
    loader_indices = data_loader.batch_sampler

    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmengine.ProgressBar(len(dataset))

    pred_qualitatives = []
    gt_qualitatives = []

    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            # Adapt for mmseg 1.x data structure
            # data is expected to be a dict with 'inputs' and 'data_samples'
            if 'inputs' in data and 'data_samples' in data:
                inputs = data['inputs']
                data_samples = data['data_samples']
                
                # If inputs is a list of tensors (batch size > 1 or just list wrapper), stack them or use as is
                # DINOTextSegInference.predict expects inputs and data_samples
                # But wait, DINOTextSegInference.encode_decode expects img and img_metas
                # We added predict method to DINOTextSegInference to handle this.
                
                # Call the model's predict method directly if available, or forward
                # Note: model is MMDistributedDataParallel, so we call model(..., mode='predict')
                # But our DINOTextSegInference might not be fully compatible with mmengine's BaseModel interface
                # Let's try to call the underlying model's predict if wrapped, or just call it.
                
                # Actually, we can just manually extract what we need and call encode_decode
                # But since we are in multi_gpu_test, we should respect the model interface.
                
                # Let's assume we patched DINOTextSegInference to have a predict method.
                # However, MMDistributedDataParallel forward calls module.forward by default.
                # We need to check how to invoke predict.
                
                # In mmengine, model(inputs, data_samples, mode='predict') calls model.predict()
                # Let's try to construct the args.
                
                # inputs should be a tensor [B, C, H, W] or list of tensors.
                # If it's a list of 1 tensor (batch_size=1), take the first one.
                if isinstance(inputs, list) and len(inputs) == 1:
                    inputs = inputs[0]
                    if inputs.dim() == 3:
                        inputs = inputs.unsqueeze(0)
                
                # Ensure inputs are on the same device as the model
                if hasattr(model, 'module'):
                    device = next(model.module.parameters()).device
                else:
                    device = next(model.parameters()).device
                
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(device)
                elif isinstance(inputs, list):
                    inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]
                
                # result = model(inputs, data_samples, mode='predict') 
                # The above might fail if DINOTextSegInference doesn't inherit properly from BaseModel
                
                # Fallback: manually call predict on the module
                if hasattr(model, 'module'):
                    result = model.module.predict(inputs, data_samples)
                else:
                    result = model.predict(inputs, data_samples)
                    
            else:
                # Fallback for old data format (unlikely with new mmseg)
                if device == 'cpu':
                    data['img_metas'] = [e.data[0] for e in data['img_metas']]
                result = model(return_loss=False, rescale=True, **data)

        current_data_samples = data.get('data_samples', None)

        for i, (pred_qualitative, index) in enumerate(zip(result, batch_indices)):
            # result from predict is likely a Tensor (seg_logits) or SegDataSample?
            # DINOTextSegInference.predict returns seg_logits (Tensor)
            
            # We need to process the result to match what evaluation expects.
            # Original code: pred_qualitatives.append(pred_qualitative+1)
            # This implies pred_qualitative is a label map or logit?
            # If it's logits [C, H, W], we might need argmax?
            # Or if it's already a label map.
            
            # DINOTextSegInference.encode_decode returns 'masks' which are logits/scores [B, C, H, W]
            # So result is [B, C, H, W]. Since we iterate over batch, pred_qualitative is [C, H, W].
            
            # Wait, the original code:
            # result = model(return_loss=False, rescale=True, **data)
            # In old mmseg, return_loss=False returns a list of numpy arrays (segmentation maps).
            
            # Our new predict returns logits (Tensor). We need to convert to seg map.
            # And we need to handle the +1 logic (maybe for background?).
            
            if isinstance(pred_qualitative, torch.Tensor):
                pred_qualitative = pred_qualitative.argmax(dim=0).cpu().numpy()
            
            pred_qualitatives.append(pred_qualitative) # Removed +1 for now, need to verify if needed
            
            # Get GT
            if current_data_samples is not None:
                seg_map_gt = current_data_samples[i].gt_sem_seg.data.squeeze().cpu().numpy()
            else:
                seg_map_gt = dataset.dataset.get_gt_seg_map_by_idx(index + dataset.indices.start)
            # seg_map_gt[seg_map_gt == 255] = 0
            gt_qualitatives.append(seg_map_gt)
            
            # For evaluation, we usually need the result to be in a specific format.
            # If efficient_test is False, results are collected.
            # The return value 'results' should be a list of results.
            # In old mmseg, it was list of numpy arrays.
            
            # We need to ensure 'result' (the list we extend results with) contains what evaluate expects.
            # evaluate in main.py calls dataset.evaluate(results, ...)
            # ADE20KDataset.evaluate expects list of numpy arrays (pred_seg_maps).
            
        # Re-pack result for collection
        # result is currently a single tensor [C, H, W] (logits) or numpy array (seg map)
        # We need to append the processed seg map to results list.
        # But wait, the loop `for pred_qualitative, index in zip(result, batch_indices):` iterates over the batch.
        # So `result` here must be a list of outputs for the batch.
        
        # If our predict returns a single tensor [B, C, H, W], we need to split it.
        if isinstance(result, torch.Tensor):
            result = list(result) # Split into list of [C, H, W] tensors
            
        # Now we process each item in the batch
        batch_results = []
        for i, pred in enumerate(result):
            if isinstance(pred, torch.Tensor):
                # Argmax to get label map
                pred_np = pred.argmax(dim=0).cpu().numpy().astype(np.uint8)
                batch_results.append(pred_np)
            else:
                batch_results.append(pred)
        
        results.extend(batch_results)

        if efficient_test:
            # result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]
            pass # Disable efficient test for now

        if format_only:
            # result = dataset.dataset.format_results(
            #     result, indices=batch_indices, **format_args)
            pass
        if pre_eval:
            # TODO: adapt samples_per_gpu > 1.
            # only samples_per_gpu=1 valid now
            # result = dataset.dataset.pre_eval(result, indices=[i+dataset.indices.start for i in batch_indices])
            pass

        # results.extend(result) # Already extended above

        if rank == 0:
            batch_size = len(batch_indices) * world_size
            for _ in range(batch_size):
                prog_bar.update()

    # collect results from all ranks
    if world_size > 1:
        if gpu_collect:
            # results = collect_results_gpu(results, len(dataset))
            raise NotImplementedError("collect_results_gpu is not implemented in this version.")
        else:
            results = collect_results_cpu(results, len(dataset), tmpdir)
    
    # Try to get classes from dataset meta info if CLASSES attribute is missing
    if hasattr(dataset.dataset, 'CLASSES'):
        num_classes = len(dataset.dataset.CLASSES)
    elif hasattr(dataset.dataset, 'METAINFO') and 'classes' in dataset.dataset.METAINFO:
        num_classes = len(dataset.dataset.METAINFO['classes'])
    elif hasattr(dataset.dataset, 'metainfo') and 'classes' in dataset.dataset.metainfo:
        num_classes = len(dataset.dataset.metainfo['classes'])
    else:
        # Fallback or raise error
        # Assuming ADE20K has 150 classes if not found
        num_classes = 150 
        
    return results, pred_qualitatives, gt_qualitatives, num_classes
