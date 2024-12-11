from mmengine.registry import LOOPS, HOOKS
from mmengine.runner import BaseLoop, TestLoop
from mmengine.logging import HistoryBuffer
from mmengine.hooks import Hook
from typing import Dict, List, Optional, Sequence, Tuple, Union
from mmengine.evaluator import Evaluator

from torch.utils.data import DataLoader
import torch

from mmengine.runner.amp import autocast
from mmengine.structures import BaseDataElement
from mmengine.utils import is_list_of

from ...models.classifiers.multi_domain_classifier import MultiDomainClassifier


import os

import shutil

CLASS_NAMES = ['Ring', 'Trophozoite', 'Schizont', 'Gametocyte', 
               'HealthyRBC', 'Other', 'Difficult']
DOMAIN_NAMES = ['OurPlasmodium', 'BBBC041', 'IMLMalaria']

@LOOPS.register_module()
class MultiDomainTestLoop(TestLoop):
    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 dataloaders_multi_domain: List[Union[DataLoader, Dict]] = [],
                 domain_names: List = None, #List of domain names according to dataloaders
                 save_false_cases_name: bool = False,
                 fp16: bool = False) -> None:
        
        super().__init__(runner, dataloader, evaluator)

        self.n_domains = len(dataloaders_multi_domain)
        self.dataloaders = [] #List of all dataloaders

        #build dataloaders
        for dataloader in dataloaders_multi_domain:
            if isinstance(dataloader, dict):
                self.dataloaders.append(runner.build_dataloader(dataloader, seed=runner.seed))
            else:
                self.dataloaders.append(dataloader)
        
        if domain_names is None: 
            self.domain_names = DOMAIN_NAMES
        else:
            self.domain_names = domain_names

        self.save_false_cases_name = save_false_cases_name

    def run(self) -> dict:
        """Launch test."""
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()

        # clear test loss
        self.test_loss.clear()

        #Khanh implementation for multi-domain evaluation 
        metrics_all = {}
        for domain_idx, dataloader in enumerate(self.dataloaders):
            for idx, data_batch in enumerate(dataloader):
                self.run_iter(idx, data_batch, domain_idx)

            # compute metrics
            metrics = self.evaluator.evaluate(len(dataloader.dataset))

            for metric_name in metrics.keys():
                metrics_all['{}/{}'.format(self.domain_names[domain_idx], metric_name)] = metrics[metric_name]

        if self.test_loss:
            loss_dict = _parse_losses(self.test_loss, 'test')
            metrics.update(loss_dict)
        
        #Copying false classification cases
        metrics_all = self.save_false_classification(metrics_all)
                                    
        self.runner.call_hook('after_test_epoch', metrics=metrics_all)
        self.runner.call_hook('after_test')
        return metrics
    
    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict],
                 domain_idx: int) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)
        # predictions should be sequence of BaseDataElement

        with autocast(enabled=self.fp16):
            if isinstance(self.runner.model, MultiDomainClassifier):
                outputs = self.runner.model.test_step(data_batch, domain_idx = domain_idx)
            else: 
                outputs = self.runner.model.test_step(data_batch)

        outputs, self.test_loss = _update_losses(outputs, self.test_loss)

        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_test_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)



    def save_false_classification(self, metrics_all):
        for domain_idx in range(len(self.dataloaders)):
            metric_name = '{}/{}'.format(self.domain_names[domain_idx], 'false_classification')
            
            if metric_name in metrics_all.keys():
                for gt in metrics_all[metric_name]:
                    for pred in metrics_all[metric_name][gt]:
                        path = os.path.join(self.runner.work_dir, 'false_classification', 
                                            self.domain_names[domain_idx], CLASS_NAMES[gt], 
                                            CLASS_NAMES[pred])
                        os.makedirs(path, exist_ok=True)

                        for i in range(len(metrics_all[metric_name][gt][pred])):
                            # print(metrics_all[metric_name][gt][pred][i])
                            # print(os.path.join(path, '{}.jpg'.format(i)))

                            img_path = metrics_all[metric_name][gt][pred][i]
                            shutil.copy(img_path, os.path.join(path, '{}.jpg'.format(i)))        
                
                            if (gt == pred) and (gt == 4) and (i == 30):
                                break
                        
                        if gt == pred:
                            metrics_all[metric_name][gt][pred] = []

                # If original cropped cells are not needed
                if not self.save_false_cases_name:
                    metrics_all.pop(metric_name)
        return metrics_all

def _update_losses(outputs: list, losses: dict) -> Tuple[list, dict]:
    """Update and record the losses of the network.

    Args:
        outputs (list): The outputs of the network.
        losses (dict): The losses of the network.

    Returns:
        list: The updated outputs of the network.
        dict: The updated losses of the network.
    """
    if isinstance(outputs[-1],
                  BaseDataElement) and outputs[-1].keys() == ['loss']:
        loss = outputs[-1].loss  # type: ignore
        outputs = outputs[:-1]
    else:
        loss = dict()

    for loss_name, loss_value in loss.items():
        if loss_name not in losses:
            losses[loss_name] = HistoryBuffer()
        if isinstance(loss_value, torch.Tensor):
            losses[loss_name].update(loss_value.item())
        elif is_list_of(loss_value, torch.Tensor):
            for loss_value_i in loss_value:
                losses[loss_name].update(loss_value_i.item())
    return outputs, losses

def _parse_losses(losses: Dict[str, HistoryBuffer],
                  stage: str) -> Dict[str, float]:
    """Parses the raw losses of the network.

    Args:
        losses (dict): raw losses of the network.
        stage (str): The stage of loss, e.g., 'val' or 'test'.

    Returns:
        dict[str, float]: The key is the loss name, and the value is the
        average loss.
    """
    all_loss = 0
    loss_dict: Dict[str, float] = dict()

    for loss_name, loss_value in losses.items():
        avg_loss = loss_value.mean()
        loss_dict[loss_name] = avg_loss
        if 'loss' in loss_name:
            all_loss += avg_loss

    loss_dict[f'{stage}_loss'] = all_loss
    return loss_dict