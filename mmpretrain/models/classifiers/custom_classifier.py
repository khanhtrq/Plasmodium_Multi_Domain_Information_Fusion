from typing import List, Optional
import torch
import torch.nn as nn
from mmpretrain.structures import DataSample

from .image import ImageClassifier
from mmpretrain.registry import MODELS

@MODELS.register_module()
class CustomClassifier(ImageClassifier):
    def __init__(self,
                 backbone: dict,
                 neck: Optional[dict] = None,
                 head: Optional[dict] = None,
                 discrepancy_head: Optional[dict] = None,
                 pretrained: Optional[str] = None,
                 train_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):

      super(CustomClassifier, self).__init__(backbone, neck, head, pretrained, train_cfg, data_preprocessor, init_cfg)

      #Classification head
      self.heads = []
      self.heads.append(self.head)

      if discrepancy_head is not None and not isinstance(discrepancy_head, nn.Module):
          print("INITIALIZING DOMAIN DISCREPANCY HEAD")
          discrepancy_head = MODELS.build(discrepancy_head)
          self.discrepancy_head = discrepancy_head 
          self.heads.append(self.discrepancy_head)
      
      #Possible extension of other heads    
    
    def forward(self,
              inputs: torch.Tensor,
              data_samples: Optional[List[DataSample]] = None,
              mode: str = 'tensor'):

      print("SHAPE OF INPUT:", inputs.shape)
      if mode == 'tensor':
          feats = self.extract_feat(inputs)
          return self.head(feats) if self.with_head else feats
      elif mode == 'loss':
          return self.loss(inputs, data_samples)
      elif mode == 'predict':
          return self.predict(inputs, data_samples)
      else:
          raise RuntimeError(f'Invalid mode "{mode}".')
    
    def loss(self, inputs: torch.Tensor,
             data_samples: List[DataSample]) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample]): The annotation data of
                every samples.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        feats = self.extract_feat(inputs)

        losses = {}
        
        for head in self.heads:
           loss = head.loss(feats, data_samples)
           for loss_name in loss.keys():
              losses[loss_name] = loss[loss_name]
        
        print("LOSS IN CUSTOM CLASSIFIER", losses)

        return losses          
        #return self.head.loss(feats, data_samples)

      

    def extract_feat(self, inputs, stage='neck'):
      print(type(inputs))
      x = self.backbone(inputs)
      print("Length of feaure:", len(x))
      print("SHAPE OUTPUT OF BACKBONE:", x[0].shape)

      if stage == 'backbone':
        return x

      if self.with_neck:
        x = self.neck(x)
      if stage == 'neck':
        print("STAGE 0 AFTER AVG POOLING:", x[0].shape)
        print("NECK:", x[-1].shape)
        return x
      print(x[0].shape)
      assert self.with_head and hasattr(self.head, 'pre_logits'), \
          "No head or the head doesn't implement `pre_logits` method."
      return self.head.pre_logits(x)