# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseClassifier
from .hugging_face import HuggingFaceClassifier
from .image import ImageClassifier
from .timm import TimmClassifier
from .custom_classifier import CustomClassifier
from .multi_domain_classifier import MultiDomainClassifier

__all__ = [
    'BaseClassifier', 'ImageClassifier', 'TimmClassifier',
    'HuggingFaceClassifier', 'CustomClassifier', 
    'MultiDomainClassifier'
]
