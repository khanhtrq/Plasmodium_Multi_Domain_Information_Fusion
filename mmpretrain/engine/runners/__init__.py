# Copyright (c) OpenMMLab. All rights reserved.
from .retrieval_loop import RetrievalTestLoop, RetrievalValLoop
from .multi_domain_train_loop import MultiDomainTrainLoop
from .multi_domain_val_loop import MultiDomainValLoop
from .multi_domain_test_loop import MultiDomainTestLoop

__all__ = ['RetrievalTestLoop', 'RetrievalValLoop', 'MultiDomainTrainLoop', 
           'MultiDomainValLoop', 'MultiDomainTestLoop']
