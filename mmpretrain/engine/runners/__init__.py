# Copyright (c) OpenMMLab. All rights reserved.
from .retrieval_loop import RetrievalTestLoop, RetrievalValLoop
from .custom_loop import MultiDomainTrainLoop

__all__ = ['RetrievalTestLoop', 'RetrievalValLoop', 'MultiDomainTrainLoop']
