from itertools import product
from typing import List, Optional, Sequence, Union

import mmengine
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.evaluator import BaseMetric

from mmpretrain.registry import METRICS

import os

CLASS_NAMES = ['Ring', 'Trophozoite', 'Schizont', 'Gametocyte', 'HealthyRBC', 'Other', 'Difficult']


@METRICS.register_module()
class FalseClassification(BaseMetric):
    def __init__(self,
                 num_classes: Optional[int] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        
        super().__init__(collect_device, prefix)
        self.num_classes = num_classes #num_classes
    
    def process(self, data_batch, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            if 'pred_score' in data_sample:
                pred_score = data_sample['pred_score']
                pred_label = pred_score.argmax(dim=0, keepdim=True)
                self.num_classes = pred_score.size(0)
            else:
                pred_label = data_sample['pred_label']
            
            self.results.append({
                'img_path': data_sample['img_path'],
                'pred_label': pred_label,
                'gt_label': data_sample['gt_label'],
            })
        
    
    def compute_metrics(self, results: list) -> dict:
        
        false_classification = {}
        for i in range(self.num_classes):
            false_classification[i] = {}
            for j in range(self.num_classes):
                false_classification[i][j] = []
        
        for sample in results:
            gt_label = sample['gt_label'].item()
            pred_label = sample['pred_label'].item()

            false_classification[gt_label][pred_label].append(sample['img_path'])

        return {'false_classification': false_classification}
