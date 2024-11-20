# Copyright (c) OpenMMLab. All rights reserved.
from .retrieval_loop import RetrievalTestLoop, RetrievalValLoop
from .multi_domain_train_loop import MultiDomainTrainLoop
from .multi_domain_val_loop import MultiDomainValLoop

__all__ = ['RetrievalTestLoop', 'RetrievalValLoop', 'MultiDomainTrainLoop', 
           'MultiDomainValLoop']
