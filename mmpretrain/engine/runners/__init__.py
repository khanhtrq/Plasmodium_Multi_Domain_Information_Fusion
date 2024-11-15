# Copyright (c) OpenMMLab. All rights reserved.
from .retrieval_loop import RetrievalTestLoop, RetrievalValLoop
from .custom_loop import CustomTrainLoop

__all__ = ['RetrievalTestLoop', 'RetrievalValLoop', 'CustomTrainLoop']
