from itertools import product
from typing import List, Optional, Sequence, Union

import mmengine
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.evaluator import BaseMetric

from mmpretrain.registry import METRICS


@METRICS.register_module()
class FalseClassification(BaseMetric):
    def __init__(self,
                num_classes: Optional[int] = None,
                collect_device: str = 'cpu',
                prefix: Optional[str] = None) -> None:
        
        super().__init__(collect_device, prefix)
        self.num_classes = 6 #num_classes
    
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

            print(data_sample)
            break
        
    
    def compute_metrics(self, results: list) -> dict:
        
        false_classification = np.empty((self.num_classes, self.num_classes), dtype=object)

        for sample in results:
            if sample['gt_label'] != sample['pred_label']:
                if false_classification[sample['gt_label'], sample['pred_label']] is None:
                    false_classification[sample['gt_label'], sample['pred_label']] = [sample['img_path']]
                else:      
                    false_classification[sample['gt_label'], sample['pred_label']].append(sample['img_path'])            


        return {'false_classification': false_classification}
