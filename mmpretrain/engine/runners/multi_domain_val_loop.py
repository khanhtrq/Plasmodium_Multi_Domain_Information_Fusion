from mmengine.registry import LOOPS, HOOKS
from mmengine.runner import BaseLoop, ValLoop
from mmengine.logging import HistoryBuffer
from mmengine.hooks import Hook
from typing import Dict, List, Optional, Sequence, Tuple, Union
from mmengine.evaluator import Evaluator

from torch.utils.data import DataLoader
import torch

@LOOPS.register_module()
class MultiDomainValLoop(ValLoop):
    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 dataloaders_multi_domain: List[Union[DataLoader, Dict]] = [],
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

    
    def run(self) -> dict:
        """Launch validation."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()

        # clear val loss
        self.val_loss.clear()

        metrics_all = {}
        for idx_domain, dataloader in enumerate(self.dataloaders):
            # print("ONE DOMAIN IS EVALUATED", idx_domain)
            for idx, data_batch in enumerate(dataloader):
                self.run_iter(idx, data_batch)
                # print("Done with one iteration ValLoop!")
                # break

            # compute metrics
            metrics = self.evaluator.evaluate(len(dataloader.dataset))
            # print("TYPE OF metrics:", type(metrics))
            # print("KEYS IN metrics:", metrics.keys())
            # print("METRICS:",  metrics)

            for metric_name in metrics.keys():
                # print(metric_name)
                # print('{}_domain{}'.format(metric_name, idx_domain+1))
                # print('domain{}/{}'.format(idx_domain, metric_name))
                metrics_all['domain{}/{}'.format(idx_domain + 1, metric_name)] = metrics[metric_name]

        # metrics = {'dumb_metric': -1}
        # print(metrics_all)
        
        # ValLoop implementation
        if self.val_loss:
            loss_dict = _parse_losses(self.val_loss, 'val')
            metrics.update(loss_dict)

        self.runner.call_hook('after_val_epoch', metrics=metrics_all)
        self.runner.call_hook('after_val')

        return metrics_all

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