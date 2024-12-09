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
    def __init__(self, input_dim, hidden_dim, output_dim,
                 n_domains, ema_alpha = 0.6):

        super(MultiDomainInformationFusion, self).__init__()

        self.gap = GlobalAveragePooling()

        self.gcn_conv1 = GCNConv(input_dim, hidden_dim)  # First GCN layer
        self.gcn_conv2 = GCNConv(hidden_dim, output_dim)  # Second GCN layer

        self.weight_head = nn.Linear(input_dim, input_dim)

        self.n_domains = n_domains

        self.agent_node_ema = nn.Parameter(torch.zeros(n_domains, input_dim),
                                           requires_grad = False)
        if torch.cuda.is_available():
            self.agent_node_ema.to(device='cuda')
        self.ema_alpha = ema_alpha

        '''
        self.agent_nodes: Tensor, shape (self.n_domains, n_agent_nodes = n_classes)
        batch_size: information to know which samples are from which domain
        self.wieght_head: Conv 1x1, head to get contribution weight for agent nodes

        '''
        

    def forward(self, instance_node: torch.Tensor,
                data_samples = None,
                mode: str = 'loss',
                domain_idx: int = None):
        """
        Input:
            instance_node: list of features from multiple levels, default: last layer backbone
        Args:
            instance_node: Node feature matrix (num_nodes instance_node input_dim)
            mode (str): 'loss' for training, 'predict' for inference
        Return:
            Information-fused features (without agent node features)
        """

        instance_node = instance_node[0]

        # print("MODE IN NECK FROM CLASSIFIER:", mode)


        #Global Average Pooling
        instance_node = self.gap(instance_node)
        instance_node = instance_node.view(instance_node.size(0), -1)

        if mode == 'loss':
            first_edge_indicies, second_edge_indicies = self.domain_graph_training(instance_node)
            
            agent_node = self.agent_node(instance_node)
            self.update_agent_node(agent_node)
        elif mode =='predict':
            # print('PREIDCT PHASE IN MULTI DOMAIN NECK')
            first_edge_indicies, second_edge_indicies = self.domain_graph_inference(instance_node, 
                                                            domain_idx= domain_idx)
            pass
        
        print("DEVICE OF TENSOR:", self.gcn_conv1.lin.weight.device, instance_node.device, self.agent_node_ema.device)
        
        node_1st = self.gcn_conv1(torch.cat((instance_node, self.agent_node_ema), dim = 0), 
                                       first_edge_indicies)  # Apply first GCN layer
        
        node_2nd = self.gcn_conv2(node_1st, second_edge_indicies)  # Apply second GCN layer        
        #Update agents node 
        
        instance_node = node_2nd[:-self.n_domains]

        return tuple([instance_node])
    
    def agent_node(self, instance_node: torch.Tensor):
        '''
        input: instance features 
        output: agent nodes for current iteration

        Calculate weighted agent nodes by classes

        instance_node --> weight_head --> instance weight
        '''

        # Returns a new Tensor, detached from the current graph.
        instance_node = instance_node.detach()
        batch_size = instance_node.shape[0] // self.n_domains
        
        weight = self.weight_head(instance_node)
        # Shape of weight: (Batch_size * n_domain, feat_dim)
        # Expected reshaped: (self.n_domains, batch_size, feat_dim)


        weight = weight.view(self.n_domains, batch_size, -1)
        instance_node = instance_node.view(self.n_domains, batch_size, -1)
        
        #Normalize along batch dim, so that the sum equals to 1
        weight = F.softmax(weight, dim = 1)

        # print("SHAPE OF WEIGHT AND INSTANCE FEATURE:", weight.shape, instance_node.shape)

        
        #Dot product on batch dimension, sum over batch dimension
        agent_node = (weight*instance_node).sum(dim = 1)
        # print('SHAPE OF AGENT NODE FEATURE:',  agent_node.shape)
        # print("SHAPE OF WIEGHT:", weight.shape)

        return agent_node
    
    def domain_graph_training(self, instance_node: torch.Tensor):
        batch_size = instance_node.shape[0] // self.n_domains
        first_kn_graph = torch.zeros(instance_node.shape[0] + self.n_domains, 
                                     instance_node.shape[0] + self.n_domains)
        second_kn_graph = torch.zeros(instance_node.shape[0] + self.n_domains, 
                                     instance_node.shape[0] + self.n_domains)
        
        first_kn_graph[instance_node.shape[0]:, 
                       instance_node.shape[0]:] = 1
        second_kn_graph[instance_node.shape[0]:, 
                       instance_node.shape[0]:] = 1

        for idx_domain in range(self.n_domains):
            first_kn_graph[instance_node.shape[0] + idx_domain, 
                           instance_node.shape[0] + idx_domain] = 0
            second_kn_graph[instance_node.shape[0] + idx_domain, 
                           instance_node.shape[0] + idx_domain] = 0

            # instance node --> agent node
            second_kn_graph[batch_size*idx_domain : batch_size * (idx_domain+1),
                            -self.n_domains + idx_domain] = 1
            second_kn_graph[-self.n_domains + idx_domain,
                            batch_size*idx_domain : batch_size * (idx_domain+1)] = 1
        
        # print(second_kn_graph)
        # print(first_kn_graph)
        # print("SECOND KNOWLEDGE GRAPH:\n", second_kn_graph.nonzero(as_tuple=False).t())
        
        first_edge_indicies = first_kn_graph.nonzero(as_tuple=False).t()
        second_edge_indicies = second_kn_graph.nonzero(as_tuple=False).t()

        return first_edge_indicies, second_edge_indicies
    
    def domain_graph_inference(self, instance_node: torch.Tensor, domain_idx):
        first_kn_graph = torch.zeros(instance_node.shape[0] + self.n_domains, 
                                     instance_node.shape[0] + self.n_domains)
        second_kn_graph = torch.zeros(instance_node.shape[0] + self.n_domains, 
                                     instance_node.shape[0] + self.n_domains)
        #Agent node connection
        first_kn_graph[instance_node.shape[0]:, 
                       instance_node.shape[0]:] = 1
        second_kn_graph[instance_node.shape[0]:, 
                       instance_node.shape[0]:] = 1

        for idx_domain in range(self.n_domains):
            first_kn_graph[instance_node.shape[0] + idx_domain, 
                           instance_node.shape[0] + idx_domain] = 0
            second_kn_graph[instance_node.shape[0] + idx_domain, 
                           instance_node.shape[0] + idx_domain] = 0
        
        #Instance node to agent node connection
        second_kn_graph[:instance_node.shape[0], -self.n_domains + domain_idx ] = 1
        second_kn_graph[-self.n_domains + domain_idx, :instance_node.shape[0]] = 1

        # print("ADJACENCY MATRIX INFERENCE DOMAIN INDEX {}:".format(domain_idx))        
        # print(first_kn_graph)
        # print(second_kn_graph)

        first_edge_indicies = first_kn_graph.nonzero(as_tuple=False).t()
        second_edge_indicies = second_kn_graph.nonzero(as_tuple=False).t()

        return first_edge_indicies, second_edge_indicies


    def update_agent_node(self, agent_node):
        if torch.all(self.agent_node_ema == 0):
            #Intialize torch tensor
            # print(self.agent_node_ema)
            self.agent_node_ema.add_(agent_node.detach())
            # print("After:", self.agent_node_ema)
            # print(agent_node)
        else:
            self.agent_node_ema.data = (1-self.ema_alpha)*self.agent_node_ema.data + self.ema_alpha*agent_node
            # print("TYPE OF AGENT NODE EMA:", type(self.agent_node_ema))       
            # print("Neck model dictioinary:", self.state_dict().keys()) 
        '''
        To dos:
            - Do not update in evaluation mode: DONE
            - Save and load agent node value to evaluate during test phase: DONE
                By defining it as model parameters with requires_grad = False
        '''
        # self.agent_node_ema = agent_node
    
        