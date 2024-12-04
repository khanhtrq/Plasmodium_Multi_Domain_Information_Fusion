from mmengine.registry import LOOPS, HOOKS
from mmengine.runner import BaseLoop, EpochBasedTrainLoop
from mmengine.hooks import Hook
from typing import Dict, List, Optional, Sequence, Tuple, Union

from torch.utils.data import DataLoader
import torch

from itertools import chain



# Customized validation loop
@LOOPS.register_module()
class MultiDomainTrainLoop(EpochBasedTrainLoop):

  def __init__(self, 
                runner, 
                dataloader: Union[DataLoader, Dict], 
                max_epochs: int, 
                dataloaders_multi_domain: List[Union[DataLoader, Dict]] = [], 
                dataloader1: Union[DataLoader, Dict] = None,
                val_begin: int = 1, 
                val_interval: int = 1, 
                norm_coefficient: bool = False,
                dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
    
    super().__init__(runner, dataloader, max_epochs, val_begin, val_interval, dynamic_intervals)

    self.n_domains = len(dataloaders_multi_domain)
    self.dataloaders = [] #List of all dataloaders
    self.min_dataloader_len = float('inf')

    self.norm_coefficient = norm_coefficient

    #build dataloaders
    for dataloader in dataloaders_multi_domain:
      if isinstance(dataloader, dict):
        self.dataloaders.append(runner.build_dataloader(dataloader, seed=runner.seed))
      else:
        self.dataloaders.append(dataloader)

    for dataloader in self.dataloaders:
      if len(dataloader) < self.min_dataloader_len:
        self.min_dataloader_len = len(dataloader)
  

  def run_epoch(self) -> None:

        if self.norm_coefficient:
          self.normalization_coefficients()
          self.runner.call_hook('after_train_epoch')
          self._epoch += 1

          return

        """Iterate one epoch."""
        self.runner.call_hook('before_train_epoch')
        self.runner.model.train()

        #dataloader --> iterable dataloader
        dataloaders_iter = []
        for dataloader in self.dataloaders:
           dataloaders_iter.append(iter(dataloader))
        
        #for one epoch, number of iterations is defined by the smallest dataloader
        for idx in range(self.min_dataloader_len):
          data = []
          for idx_domain in range(self.n_domains):
            data.append(next(dataloaders_iter[idx_domain]))

          #data_batch['inputs']: tensor, shape [D*B, C, H, w], first dim in domain order
          #data_batch[data_samples]: list, in domain order, e.g. [d1, d2, d3]
          data_batch = {}
          data_batch['inputs'] = torch.cat([d['inputs'] for d in data], dim=0)
          data_batch['data_samples'] = []
          for d in data:
            data_batch['data_samples'].extend(d['data_samples'])
  
          self.run_iter(idx, data_batch)

        # -------------
        # Original EpochBasedTrainLoop
        # -------------
        self.runner.call_hook('after_train_epoch')
        self._epoch += 1

  def normalization_coefficients(self):

    print("NORMALIZATION")

    # Initialize variables
    total_sum = torch.zeros(3)
    total_sum_squared = torch.zeros(3)
    num_samples = 0
    num_pixels = 0

    #print("LENGHT OF DATALODER:", len(dataloader_all))

    for dataloader in self.dataloaders:
      for data in dataloader:  # Loop over the dataset
          print("Processed samples: {}".format(num_samples))
          # Batch size, channels, height, width

          images = torch.cat([d['inputs'] for d in data], dim=0)
          batch_size, channels, height, width = images.shape

          # Update pixel count
          num_samples += batch_size
          num_pixels += batch_size * height * width


          # Sum and squared sum
          total_sum += images.sum([0, 2, 3])  # Sum over batch, height, width
          total_sum_squared += (images ** 2).sum([0, 2, 3])

    # Calculate mean and std
    mean = total_sum / num_pixels
    std = (total_sum_squared / num_pixels - mean ** 2).sqrt()

    print("Total number of samples:", num_samples)

    print(f"Mean: {mean}")
    print(f"Std: {std}")

  