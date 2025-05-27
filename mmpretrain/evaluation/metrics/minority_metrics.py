from itertools import product
from typing import List, Optional, Sequence, Union

import mmengine
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.evaluator import BaseMetric
from .single_label import ConfusionMatrix

from mmpretrain.registry import METRICS

import os

CLASS_NAMES = ['Ring', 'Trophozoite', 'Schizont', 'Gametocyte', 'HealthyRBC', 'Other', 'Difficult']


@METRICS.register_module()
class MinorityMetrics(BaseMetric):
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
                'pred_label': pred_label,
                'gt_label': data_sample['gt_label'],
            })

    def compute_metrics(self, results: list) -> dict:
        pred_labels = []
        gt_labels = []
        for result in results:
            pred_labels.append(result['pred_label'])
            gt_labels.append(result['gt_label'])
        confusion_matrix = ConfusionMatrix.calculate(
            torch.cat(pred_labels),
            torch.cat(gt_labels),
            num_classes=self.num_classes)

        precision_recall_metrics = {}
        for i in range(self.num_classes):
            true_positives = confusion_matrix[i, i]
            false_positives = sum(confusion_matrix[:, i]) - true_positives
            false_negatives = sum(confusion_matrix[i, :]) - true_positives
            
            # Calculate precision_recall_metrics
            precision_i = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
            recall_i = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
            
            precision_recall_metrics['precision/{}'.format(CLASS_NAMES[i])] = precision_i
            precision_recall_metrics['recall/{}'.format(CLASS_NAMES[i])] = recall_i
        
        parasitized_correct = 0
        parasitized_recall = 0
        parasitized_precision  = 0

        for i in range(4):
            parasitized_correct += confusion_matrix[i, i]
            parasitized_recall += sum(confusion_matrix[i, :])
            parasitized_precision += sum(confusion_matrix[:, i])
        precision_recall_metrics['precision/parasitized'] = parasitized_correct / parasitized_precision
        precision_recall_metrics['recall/parasitized'] = parasitized_correct / parasitized_recall

        return precision_recall_metrics
    
    @staticmethod
    def compute_metrics_from_matrix(confusion_matrix, num_classes):
        precision_recall_metrics = {}
        for i in range(num_classes):
            true_positives = confusion_matrix[i, i]
            false_positives = sum(confusion_matrix[:, i]) - true_positives
            false_negatives = sum(confusion_matrix[i, :]) - true_positives
            
            # Calculate precision_recall_metrics
            precision_i = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
            recall_i = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
            
            precision_recall_metrics['precision/{}'.format(CLASS_NAMES[i])] = float(precision_i)
            precision_recall_metrics['recall/{}'.format(CLASS_NAMES[i])] = float(recall_i)
        
        parasitized_correct = 0
        parasitized_recall = 0
        parasitized_precision  = 0

        for i in range(4):
            parasitized_correct += confusion_matrix[i, i]
            parasitized_recall += sum(confusion_matrix[i, :])
            parasitized_precision += sum(confusion_matrix[:, i])
        precision_recall_metrics['precision/parasitized'] = float(parasitized_correct / parasitized_precision)
        precision_recall_metrics['recall/parasitized'] = float(parasitized_correct / parasitized_recall)

        return precision_recall_metrics