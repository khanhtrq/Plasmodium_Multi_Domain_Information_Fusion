# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union, Dict

import torch
import torch.nn as nn

from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from .base import BaseClassifier
from .image import ImageClassifier


@MODELS.register_module()
class MultiDomainClassifier(ImageClassifier):

    def __init__(self,
                 backbone: dict,
                 neck: Optional[dict] = None,
                 head: Optional[dict] = None,
                 pretrained: Optional[str] = None,
                 train_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 domain_idx: int = 1 # IML Malaria
                 ):
        
        super().__init__(backbone, neck, head,
                        pretrained, train_cfg, data_preprocessor,
                        init_cfg)
        self.domain_idx = domain_idx
        
    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[DataSample]] = None,
                mode: str = 'tensor',
                domain_idx: int = None):
        
        if domain_idx is None:
            domain_idx = self.domain_idx

        if mode == 'tensor':
            feats = self.extract_feat(inputs, mode)
            return self.head(feats) if self.with_head else feats
        elif mode == 'loss':
            return self.loss(inputs, data_samples, mode)
        elif mode == 'predict':
            # assert domain index
            return self.predict(inputs, domain_idx, data_samples, mode)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def extract_feat(self, inputs, mode, 
                     data_samples: Optional[List[DataSample]] = None, 
                     stage='neck',
                     domain_idx: int = None):
        assert stage in ['backbone', 'neck', 'pre_logits'], \
            (f'Invalid output stage "{stage}", please choose from "backbone", '
             '"neck" and "pre_logits"')

        x = self.backbone(inputs)

        if stage == 'backbone':
            return x
        # print('MODE IN EXTRACT FEATURE FUNCTION:', mode)
        if self.with_neck:
            x = self.neck(x, mode=mode, domain_idx = domain_idx,
                          data_samples = data_samples)
        if stage == 'neck':
            return x

        assert self.with_head and hasattr(self.head, 'pre_logits'), \
            "No head or the head doesn't implement `pre_logits` method."
        return self.head.pre_logits(x)

    def loss(self, inputs: torch.Tensor,
             data_samples: List[DataSample],
             mode: str) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample]): The annotation data of
                every samples.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        feats = self.extract_feat(inputs, mode=mode, data_samples=data_samples)
        return self.head.loss(feats, data_samples)
    

    def predict(self,
                inputs: torch.Tensor,
                domain_idx: int, 
                data_samples: Optional[List[DataSample]] = None,
                mode: str = 'predict',
                **kwargs) -> List[DataSample]:
        """Predict results from a batch of inputs.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample], optional): The annotation
                data of every samples. Defaults to None.
            **kwargs: Other keyword arguments accepted by the ``predict``
                method of :attr:`head`.
        """
        # print("MODE IN CLASSIFIER:", mode)
        feats = self.extract_feat(inputs, mode=mode, domain_idx= domain_idx,
                                  data_samples=data_samples)
        return self.head.predict(feats, data_samples, **kwargs)
    
    #Override from BaseModel
    def val_step(self, data: Union[tuple, dict, list],
                 domain_idx: int) -> list:
        """Gets the predictions of given data.

        Calls ``self.data_preprocessor(data, False)`` and
        ``self(inputs, data_sample, mode='predict')`` in order. Return the
        predictions which will be passed to evaluator.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        data = self.data_preprocessor(data, False)
        return self._run_forward(data, mode='predict', domain_idx= domain_idx)  # type: ignore
    
    def test_step(self, data: Union[dict, tuple, list],
                  domain_idx: int = None) -> list:
        """``BaseModel`` implements ``test_step`` the same as ``val_step``.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        data = self.data_preprocessor(data, False)
        if domain_idx is None:
            domain_idx = self.domain_idx
            
        return self._run_forward(data, mode='predict', domain_idx= domain_idx)  # type: ignore


    def _run_forward(self, data: Union[dict, tuple, list],
                     mode: str,
                     domain_idx: int = None) -> Union[Dict[str, torch.Tensor], list]:
        """Unpacks data for :meth:`forward`

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            mode (str): Mode of forward.

        Returns:
            dict or list: Results of training or testing mode.
        """
        if isinstance(data, dict):
            results = self(**data, mode=mode, domain_idx = domain_idx)
        elif isinstance(data, (list, tuple)):
            results = self(*data, mode=mode, domain_idx = domain_idx)
        else:
            raise TypeError('Output of `data_preprocessor` should be '
                            f'list, tuple or dict, but got {type(data)}')
        return results
