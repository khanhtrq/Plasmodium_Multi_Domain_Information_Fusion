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
                a: int,
                dataloader: Union[DataLoader, Dict], 
                max_epochs: int, 
                dataloader1: Union[DataLoader, Dict] = None,
                val_begin: int = 1, 
                val_interval: int = 1, 
                dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
    
    super().__init__(runner, dataloader, max_epochs, val_begin, val_interval, dynamic_intervals)


    if isinstance(dataloader1, dict):
      # Determine whether or not different ranks use different seed.
      diff_rank_seed = runner._randomness_cfg.get(
          'diff_rank_seed', False)
      self.dataloader1 = runner.build_dataloader(
          dataloader1, seed=runner.seed, diff_rank_seed=diff_rank_seed)
    else:
      self.dataloader1 = dataloader1

    print("TYPE OF TWO DATA LOADERS:")
    print(type(self.dataloader))
    print(type(self.dataloader1))

  def run_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.call_hook('before_train_epoch')
        self.runner.model.train()

        print("Length of dataloaders:", len(self.dataloader), len(self.dataloader1))

        for idx, (data_batch1, data_batch2) in enumerate(zip(self.dataloader, self.dataloader1)):
        
          print("TYPE OF TWO DATA INPUTS:")
          print(type(data_batch1["inputs"]))
          print(type(data_batch2["inputs"]))

          print("TYPE OF TWO DATA SAMPLES:")
          print(type(data_batch1['data_samples']), 'length:', len(data_batch1['data_samples']))
          print(type(data_batch2['data_samples']), 'length:', len(data_batch1['data_samples']))

          print('EXAMPLE OF DATA SAMPLES:')
          print(data_batch1['data_samples'][0])

          print("Shape of input data:", data_batch1['inputs'].shape)
          
          data_batch = {}
          #data_batch['inputs'] = data_batch1['inputs'] + data_batch2['inputs']
          data_batch['inputs'] = torch.cat((data_batch1['inputs'], data_batch2['inputs']), dim= 0)

          #Concatenate data from multiple dataloader
          data_batch['data_samples'] = data_batch1['data_samples'] + data_batch2['data_samples']

          print("Shape of concatenated data:", data_batch['inputs'].shape)
          print("Length of data samples: ", len(data_batch['data_samples']))
          print("--------------------------")
          print("TYPE OF DATA SAMPLES: ", type(data_batch1['data_samples']))

          #Input to model to process
          self.run_iter(idx, data_batch)
          print("Done one iter with concatenated inputs")


        # for idx, data_batch in enumerate(self.dataloader1):
        #     self.run_iter(idx, data_batch)
        #     print(type(data_batch['inputs']))
        #     print("List of first dataloader could be input to model")
        #     break

        self.runner.call_hook('after_train_epoch')
        self._epoch += 1

  