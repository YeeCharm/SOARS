
import torch
import numpy as np
from mmseg.evaluation import IoUMetric
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample

def evaluate_dataset(dataset, results, logger=None):
    """
    Evaluate results using IoUMetric.
    
    Args:
        dataset: The dataset object (mmseg 1.x).
        results: List of prediction results (numpy arrays of segmentation maps).
        logger: Logger object.
        
    Returns:
        dict: Evaluation metrics.
    """
    # Create IoUMetric
    iou_metric = IoUMetric(iou_metrics=['mIoU'])
    
    # Handle Subset
    if isinstance(dataset, torch.utils.data.Subset):
        metainfo = dataset.dataset.metainfo
    else:
        metainfo = dataset.metainfo
        
    iou_metric.dataset_meta = metainfo
    
    print(f"Evaluating {len(results)} results...")
    
    for i, pred_mask in enumerate(results):
        # Load sample (now includes GT because we added LoadAnnotations)
        item = dataset[i]
        gt_data_sample = item['data_samples']
        
        # Construct prediction data_sample
        pred_sample = SegDataSample()
        pred_sample.set_metainfo(gt_data_sample.metainfo)
        pred_sample.pred_sem_seg = PixelData(data=torch.from_numpy(pred_mask)[None, ...])
        
        # Process
        # IoUMetric.process(data_batch, data_samples)
        # data_batch should contain 'data_samples' with GT.
        # data_samples should contain predictions.
        
        # In mmseg 1.x, process expects data_samples to be a list of dicts or SegDataSample objects
        # But wait, the error says: TypeError: 'SegDataSample' object is not subscriptable
        # This happens at: pred_label = data_sample['pred_sem_seg']['data'].squeeze()
        # This implies data_sample is being accessed like a dict, but it's an object.
        # mmseg's IoUMetric.process implementation might expect dicts if it tries to subscript.
        # Let's check if we can convert SegDataSample to dict.
        
        pred_sample_dict = pred_sample.to_dict()
        
        # Also need to ensure gt_data_sample is compatible if needed, but process usually uses data_batch for GT
        # The error KeyError: 'gt_sem_seg' happens in process when accessing data_sample['gt_sem_seg']
        # This means data_sample (which is pred_sample_dict) is expected to have 'gt_sem_seg' OR
        # data_batch['data_samples'] elements should have it.
        
        # In mmseg 1.x IoUMetric.process:
        # for data_sample in data_samples:
        #    pred_label = data_sample['pred_sem_seg']['data'].squeeze()
        #    label = data_sample['gt_sem_seg']['data'].squeeze().to(pred_label.device)
        
        # Wait, IoUMetric.process iterates over data_samples (the second argument).
        # It expects EACH element in data_samples to contain BOTH prediction AND ground truth!
        # This is different from some other metrics where GT comes from data_batch.
        
        # So we need to merge GT into pred_sample_dict.
        
        # gt_data_sample is likely a SegDataSample object.
        if hasattr(gt_data_sample, 'gt_sem_seg'):
             pred_sample_dict['gt_sem_seg'] = gt_data_sample.gt_sem_seg.to_dict()
        elif isinstance(gt_data_sample, dict) and 'gt_sem_seg' in gt_data_sample:
             pred_sample_dict['gt_sem_seg'] = gt_data_sample['gt_sem_seg']
        
        # We don't really need batch_data_batch for IoUMetric if GT is in data_samples
        batch_data_batch = {} 
        
        # If IoUMetric expects dicts for predictions:
        iou_metric.process(batch_data_batch, [pred_sample_dict])
        
    metrics = iou_metric.compute_metrics(iou_metric.results)
    return metrics

