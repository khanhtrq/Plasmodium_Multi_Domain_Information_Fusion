from mmengine.model import BaseModule
from mmpretrain.registry import MODELS
from torch import nn
from typing import Union, Optional, Tuple, List

from mmpretrain.structures import DataSample

import torch

@MODELS.register_module()
class DomainDiscrepancyHead(BaseModule):
    def __init__(self,
                 batch_size: int,
                 # Change the default type of loss
                 loss: dict = dict(type='CrossEntropyLoss', loss_weight=1.0),
                 init_cfg: Optional[dict] = None):
        
        super().__init__(init_cfg=init_cfg)

        if not isinstance(loss, nn.Module):
            loss = MODELS.build(loss)
        self.loss_module = loss
        self.batch_size = batch_size


    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        #Return the avearge of features for each domain

        #Get feature from last layer
        feats = feats[-1]
        n_domain = int(feats.shape[0]/self.batch_size)

        feats = feats.view(n_domain, self.batch_size, -1)
        print("SHAPE OF FEATURE IN DOMAIN DISCREPANCY HEAD:", feats.shape)
        feats = torch.mean(feats, dim=1, keepdim= False)
        print("Shape of average feature:", feats.shape)

        return feats

    def loss(self, feats: Tuple[torch.Tensor], data_samples) -> dict:

        """Calculate domain discrepancy loss.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used. The shape of every item should be
                ``(num_samples, num_classes)``.
            
            data_samples is useless, just to maintain the consistency with other heads

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        #I might have to implement the Maximum Mean Discrepancey loss by myself.

        # The part can be traced by torch.fx
        feats = self(feats)
        # The part can not be traced by torch.fx
        return {"loss_mmd": torch.tensor([1.])}