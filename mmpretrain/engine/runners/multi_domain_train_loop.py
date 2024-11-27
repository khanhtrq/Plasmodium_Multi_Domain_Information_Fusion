from mmengine.registry import LOOPS, HOOKS
from mmengine.runner import BaseLoop, EpochBasedTrainLoop
from mmengine.hooks import Hook
from typing import Dict, List, Optional, Sequence, Tuple, Union

from torch.utils.data import DataLoader
import torch



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
                dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
    
    super().__init__(runner, dataloader, max_epochs, val_begin, val_interval, dynamic_intervals)

    self.n_domains = len(dataloaders_multi_domain)
    self.dataloaders = [] #List of all dataloaders
    self.min_dataloader_len = float('inf')

    #build dataloaders
    for dataloader in dataloaders_multi_domain:
      if isinstance(dataloader, dict):
        self.dataloaders.append(runner.build_dataloader(dataloader, seed=runner.seed))
      else:
        self.dataloaders.append(dataloader)

    for dataloader in self.dataloaders:
      if len(dataloader) < self.min_dataloader_len:
        self.min_dataloader_len = len(dataloader)
    
    # print("TYPE OF DATALOADER:")
    
    # print("Min numbers of iteration: ", self.min_dataloader_len)

  def run_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.call_hook('before_train_epoch')
        self.runner.model.train()

        # print("Length of dataloaders:", len(self.dataloader), len(self.dataloader1))


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
          
          # print("Finishing looping over all dataloader!")
          # print(len(data))
          # print(type(data[0]['data_samples']))

          # print(data_batch['inputs'].shape)

          # print(len(data_batch['data_samples']))

          self.run_iter(idx, data_batch)
          # print("DONE ONE ITER WITH MULTIPLE DATALOADERS")


        # for idx, (data_batch1, data_batch2) in enumerate(zip(self.dataloader, self.dataloader1)):
          
        #   data_batch = {}
        #   #data_batch['inputs'] = data_batch1['inputs'] + data_batch2['inputs']
        #   data_batch['inputs'] = torch.cat((data_batch1['inputs'], data_batch2['inputs']), dim= 0)
        #   data_batch['inputs'] = torch.cat((data_batch1['inputs'],), dim= 0)

        #   #Concatenate data from multiple dataloader
        #   data_batch['data_samples'] = data_batch1['data_samples'] + data_batch2['data_samples']
        #   data_batch['data_samples'] = data_batch1['data_samples']

        #   print("Shape of concatenated data:", data_batch['inputs'].shape)
        #   print("Length of data samples: ", len(data_batch['data_samples']))
        #   print("--------------------------")
        #   print("TYPE OF DATA SAMPLES: ", type(data_batch1['data_samples']))

        #   #Input to model to process
        #   self.run_iter(idx, data_batch)
        #   print("Done one iter with concatenated inputs")

        #   break


        # for idx, data_batch in enumerate(self.dataloader1):
        #     self.run_iter(idx, data_batch)
        #     print(type(data_batch['inputs']))
        #     print("List of first dataloader could be input to model")
        #     break

        # -------------
        # Original EpochBasedTrainLoop
        # -------------
        self.runner.call_hook('after_train_epoch')
        self._epoch += 1

  