from typing import List, Optional
import torch
from mmpretrain.structures import DataSample

from .image import ImageClassifier
from mmpretrain.registry import MODELS

@MODELS.register_module()
class CustomClassifier(ImageClassifier):
    def __init__(self,
                 backbone: dict,
                 neck: Optional[dict] = None,
                 head: Optional[dict] = None,
                 pretrained: Optional[str] = None,
                 train_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):

      super(CustomClassifier, self).__init__(backbone, neck, head, pretrained, train_cfg, data_preprocessor, init_cfg)

      print("Initializing CustomModel!")
    
    def forward(self,
              inputs: torch.Tensor,
              data_samples: Optional[List[DataSample]] = None,
              mode: str = 'tensor'):

      print("MUST CALL EXTRACT FEATURE")
      if mode == 'tensor':
          feats = self.extract_feat(inputs)
          return self.head(feats) if self.with_head else feats
      elif mode == 'loss':
          return self.loss(inputs, data_samples)
      elif mode == 'predict':
          return self.predict(inputs, data_samples)
      else:
          raise RuntimeError(f'Invalid mode "{mode}".')

    def extract_feat(self, inputs, stage='neck'):
      print(type(inputs))
      x = self.backbone(inputs)
      print("SHAPE OUTPUT OF BACKBONE:", x[0].shape)

      if stage == 'backbone':
        return x

      if self.with_neck:
        x = self.neck(x)
      if stage == 'neck':
        return x
      print(x[0].shape)
      assert self.with_head and hasattr(self.head, 'pre_logits'), \
          "No head or the head doesn't implement `pre_logits` method."
      return self.head.pre_logits(x)