import torch.nn as nn
# from torch_geometric.nn import GCNConv
import torch.nn.functional as F

from mmpretrain.registry import MODELS
from torch_geometric.nn import GCNConv
from mmengine.model import BaseModule
from .gap import GlobalAveragePooling


import torch


@MODELS.register_module()
class MultiDomainInformationFusion(BaseModule):
    '''
    NOTE:
        Also need to customize classifier to input labels to neck.
        Inherit from classifier.image.ImageClassifier class
        Replace in ImageClassifier: 
            self.neck(x) --> self.neck(x, datasamples); 
            extract_feat(inputs) --> extract_feat(inputs, data_samples)
    '''
    def __init__(self, input_dim, hidden_dim, output_dim):

        super(MultiDomainInformationFusion, self).__init__()

        self.gap = GlobalAveragePooling()

        self.gcn_conv1 = GCNConv(input_dim, hidden_dim)  # First GCN layer
        self.gcn_conv2 = GCNConv(hidden_dim, output_dim)  # Second GCN layer

        self.weight_head = nn.Conv2d(input_dim, input_dim, kernel_size=(1, 1))

        '''
        self.agent_nodes: Tensor, shape (n_domains, n_agent_nodes = n_classes)
        self.batch_size: information to know which samples are from which domain
        self.wieght_head: Conv 1x1, head to get contribution weight for agent nodes

        '''
        

    def forward(self, instance_feat: torch.Tensor, 
                kn_graph: torch.Tensor = None,
                data_samples = None):
        """
        Input:
            instance_feat: list of features from multiple levels, default: last layer backbone
        Args:
            instance_feat: Node feature matrix (num_nodes instance_feat input_dim)
            kn_graph: adjacency matrix, either node --> node or sparse matrix. Fixed
        Procedure: 
            Calculate current agent nodes, function calculate_agent_node()
            Update agent nodes by EMA, function ema()
            Average pooling instances features to shape: (batch_size, feature_dim, 1, 1)
            Concatenate instances features with agent nodes feature
            
            Input to GCN: self.gcn_conv1 and gcn_conv2
        
        Return:
            Information-fused features (without agent node features)
        """

        print("TYPE OF INPUT GCN LAYER:", type(instance_feat))
        print('LENGTH OF X:', len(instance_feat))
        print(type(instance_feat[0]))
        print('SHAPE OF INPUT:', instance_feat[0].shape)
        instance_feat = instance_feat[0]

        instance_feat = self.gap(instance_feat)
        instance_feat = instance_feat.view(instance_feat.size(0), -1)
        print("SHAPE AFTER POOLING:", instance_feat.shape)

        self.agent_feat(instance_feat)
 
        kn_graph = torch.tensor([[0],
                           [1]], dtype=torch.long)
                
        instance_feat = self.gcn_conv1(instance_feat, kn_graph)  # Apply first GCN layer
        instance_feat = self.gcn_conv2(instance_feat, kn_graph)  # Apply second GCN layer

        print("SHAPE AFTER AVERAGE POOLING:", instance_feat.shape)
        
        #Update agents node 
        
        print("SHAPE OF OUTPUT OF GCN MULTI DOMAIN FUSION:", instance_feat.shape)
        return tuple([instance_feat])
    
    def agent_feat(self, instance_feat: torch.Tensor):
        '''
        input: instance features 
        output: agent nodes for current iteration

        Calculate weighted agent nodes by classes

        instance_feat --> weight_head --> instance weight
        '''
        
        weight = self.weight_head(instance_feat)
        print("SHAPE OF WIEGHT:", weight.shape)


    def ema():
        '''
        Update agent nodes by EMA
        '''
        pass

    def build_matrix():
        '''
        Update ajacency matrix
        '''
        pass
    
        