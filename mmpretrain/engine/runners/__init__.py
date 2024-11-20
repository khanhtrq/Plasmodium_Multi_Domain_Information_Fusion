# Copyright (c) OpenMMLab. All rights reserved.
from .retrieval_loop import RetrievalTestLoop, RetrievalValLoop
from .multi_domain_loop import MultiDomainTrainLoop

__all__ = ['RetrievalTestLoop', 'RetrievalValLoop', 'MultiDomainTrainLoop']
