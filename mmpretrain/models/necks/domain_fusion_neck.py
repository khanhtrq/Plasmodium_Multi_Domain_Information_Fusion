import torch.nn as nn
# from torch_geometric.nn import GCNConv
import torch.nn.functional as F

from mmpretrain.registry import MODELS
from torch_geometric.nn import GCNConv
from mmengine.model import BaseModule
from .gap import GlobalAveragePooling

import torch
import numpy as np
from typing import List


@MODELS.register_module()
class MultiDomainInformationFusion(BaseModule):

    def __init__(self, input_dim, hidden_dim, output_dim,
                 n_domains, n_classes:int = None, ema_alpha = 0.6):

        super(MultiDomainInformationFusion, self).__init__()

        self.gap = GlobalAveragePooling()

        self.gcn_conv1 = GCNConv(input_dim, hidden_dim)  # First GCN layer
        self.gcn_conv2 = GCNConv(hidden_dim, output_dim)  # Second GCN layer

        self.weight_head = nn.Linear(input_dim, input_dim)

        self.n_domains = n_domains
        self.n_classes = n_classes

        self.agent_node_ema = nn.Parameter(torch.zeros(n_domains, input_dim),
                                           requires_grad = False)

        self.ema_alpha = ema_alpha

    def forward(self, instance_node: torch.Tensor,
                data_samples = None,
                mode: str = 'loss',
                domain_idx: int = None):

        return self._forward_domain_level(instance_node, data_samples, mode, domain_idx)

    def _forward_domain_level(self, instance_node: torch.Tensor,
                                data_samples = None,
                                mode: str = 'loss',
                                domain_idx: int = None):
        
        instance_node = instance_node[0]

        #Global Average Pooling
        instance_node = self.gap(instance_node)
        instance_node = instance_node.view(instance_node.size(0), -1)

        if mode == 'loss':
            first_edge_indicies, second_edge_indicies = self.domain_graph_training(instance_node.shape[0])
            
            agent_node = self.agent_node_domain_level(instance_node)
            agent_node_updated = self.update_ema_agent_node_domain_level(agent_node)

        elif mode =='predict':
            first_edge_indicies, second_edge_indicies = self.domain_graph_inference(instance_node.shape[0], 
                                                            domain_idx= domain_idx)
            agent_node_updated = self.agent_node_ema.detach()
        
        if torch.cuda.is_available():
            first_edge_indicies = first_edge_indicies.to(device='cuda')
            second_edge_indicies = second_edge_indicies.to(device='cuda')
  
        node_1st = self.gcn_conv1(torch.cat((instance_node, agent_node_updated), dim = 0), 
                                       first_edge_indicies)
        
        node_2nd = self.gcn_conv2(node_1st, second_edge_indicies)       
        
        instance_node = node_2nd[:-self.n_domains]

        return tuple([instance_node])
    
    def agent_node_domain_level(self, instance_node: torch.Tensor):
        '''
        Input:
            instance_node: shape (n_instances= n_domains*batch_size, feat_dim)
        Output:
            agent_node: shape (n_domains, feat_dim)

        weight: shape (n_domains, batch_size, feat_dim), softmax by dimension 1 (batch_size)
        '''

        instance_node = instance_node.detach()
        batch_size = instance_node.shape[0] // self.n_domains

        weight = self.weight_head(instance_node)
        weight = weight.view(self.n_domains, batch_size, -1)
        weight = F.softmax(weight, dim = 1)
        instance_node = instance_node.view(self.n_domains, batch_size, -1)
        
        agent_node = (weight*instance_node).sum(dim = 1)

        return agent_node
    
    def domain_graph_training(self, n_instances):

        batch_size = n_instances // self.n_domains
        first_kn_graph = torch.zeros(n_instances + self.n_domains, 
                                     n_instances + self.n_domains)
        second_kn_graph = torch.zeros(n_instances + self.n_domains, 
                                     n_instances + self.n_domains)
        
        first_kn_graph[n_instances:, n_instances:] = 1
        second_kn_graph[n_instances:, n_instances:] = 1

        for idx_domain in range(self.n_domains):
            first_kn_graph[n_instances + idx_domain, 
                           n_instances + idx_domain] = 0
            second_kn_graph[n_instances + idx_domain, 
                           n_instances + idx_domain] = 0

            # instance node --> agent node
            second_kn_graph[batch_size*idx_domain : batch_size * (idx_domain+1),
                            -self.n_domains + idx_domain] = 1
            second_kn_graph[-self.n_domains + idx_domain,
                            batch_size*idx_domain : batch_size * (idx_domain+1)] = 1

        first_edge_indicies = first_kn_graph.nonzero(as_tuple=False).t()
        second_edge_indicies = second_kn_graph.nonzero(as_tuple=False).t()

        return first_edge_indicies, second_edge_indicies
    
    def domain_graph_inference(self, n_instances, domain_idx):
        
        first_kn_graph = torch.zeros(n_instances + self.n_domains, 
                                     n_instances + self.n_domains)
        second_kn_graph = torch.zeros(n_instances + self.n_domains, 
                                     n_instances + self.n_domains)
        #Agent node connection
        first_kn_graph[n_instances:, 
                       n_instances:] = 1
        second_kn_graph[n_instances:, 
                       n_instances:] = 1

        for idx_domain in range(self.n_domains):
            first_kn_graph[n_instances + idx_domain, 
                           n_instances + idx_domain] = 0
            second_kn_graph[n_instances + idx_domain, 
                           n_instances + idx_domain] = 0
        
        #Instance node to agent node connection
        second_kn_graph[:n_instances, -self.n_domains + domain_idx ] = 1
        second_kn_graph[-self.n_domains + domain_idx, :n_instances] = 1

        # print("ADJACENCY MATRIX INFERENCE DOMAIN INDEX {}:".format(domain_idx))        
        # print(first_kn_graph)
        # print(second_kn_graph)

        first_edge_indicies = first_kn_graph.nonzero(as_tuple=False).t()
        second_edge_indicies = second_kn_graph.nonzero(as_tuple=False).t()

        return first_edge_indicies, second_edge_indicies

    def update_ema_agent_node_domain_level(self, agent_node_domain_level: torch.Tensor):
        '''
        Input: 
            agent_node_domain_level: domain-level, shape (n_domains, feat_dim) 
        '''
        if torch.all(self.agent_node_ema == 0):
            agent_node_updated = agent_node_domain_level
        else:
            agent_node_updated = (1-self.ema_alpha)*self.agent_node_ema.detach() + \
                self.ema_alpha*agent_node_domain_level

        self.agent_node_ema = torch.nn.Parameter(agent_node_updated.detach(), 
                                                 requires_grad = False)

        return agent_node_updated 

@MODELS.register_module()
class MDIFClassLevel(BaseModule):

    def __init__(self, input_dim, hidden_dim, output_dim,
                 n_domains, n_classes:int = None, ema_alpha = 0.6):

        super(MDIFClassLevel, self).__init__()

        self.gap = GlobalAveragePooling()

        self.gcn_conv1 = GCNConv(input_dim, hidden_dim)  # First GCN layer
        self.gcn_conv2 = GCNConv(hidden_dim, output_dim)  # Second GCN layer

        self.weight_head = nn.Linear(input_dim, input_dim)

        self.n_domains = n_domains
        self.n_classes = n_classes

        self.agent_node_ema = nn.Parameter(torch.zeros(n_domains*n_classes, input_dim),
                                           requires_grad = False)

        self.ema_alpha = ema_alpha

    def forward(self, instance_node: torch.Tensor,
                data_samples = None,
                mode: str = 'loss',
                domain_idx: int = None):
        print(f"Domain index in neck {domain_idx}")
        return self._forward_class_level(instance_node, data_samples, mode, domain_idx)

    def _forward_class_level(self, instance_node: torch.Tensor,
                                data_samples = None,
                                mode: str = 'loss',
                                domain_idx: int = None):
        
        instance_node = instance_node[0]

        #Global Average Pooling
        instance_node = self.gap(instance_node)
        instance_node = instance_node.view(instance_node.size(0), -1)
        
        if mode == 'loss':
            labels = np.array([data_sample.gt_label.item() for data_sample in data_samples])
            first_edge_indicies, second_edge_indicies, second_edge_weight = \
                self.class_graph_training(instance_node, labels)
            agent_node_class_level = self.agent_node_class_level(instance_node, labels)
            agent_node_updated = self.update_agent_node_class_level(agent_node_class_level)
        elif mode =='predict':
            first_edge_indicies, second_edge_indicies, second_edge_weight = \
                        self.class_graph_inference(instance_node, domain_idx)
            
            agent_node_updated = self.agent_node_ema.detach()

        # print("LABELS OF DATA:", labels) 
        # print("AGENT NODE UPDATED:", agent_node_updated[:, :3])

        if torch.cuda.is_available():
            first_edge_indicies = first_edge_indicies.to(device='cuda')
            second_edge_indicies = second_edge_indicies.to(device='cuda')
            # if mode == 'predict':
            second_edge_weight = second_edge_weight.to(device= 'cuda')
  
        node_1st = self.gcn_conv1(torch.cat((instance_node, agent_node_updated), dim = 0), 
                                       first_edge_indicies)
        if mode =='loss':
            node_2nd = self.gcn_conv2(node_1st, second_edge_indicies, edge_weight = second_edge_weight)       
        elif mode == 'predict':
            node_2nd = self.gcn_conv2(node_1st, second_edge_indicies, edge_weight = second_edge_weight)       

        instance_node = node_2nd[:-self.n_domains*self.n_classes]

        return tuple([instance_node])
    
    def agent_node_class_level(self, instance_node: torch.Tensor,
                               labels):
        '''
        Input:
            instance_node: shape (n_instances= n_domains*batch_size, feat_dim)
            labels: corresponding labels of instance_node, np.array
        Return: (n_domians, n_classes, feat_dim) 
            consider shape order to fit with adjaceny graph
        '''
        instance_node = instance_node.detach()

        n_instances = instance_node.shape[0]
        batch_size = n_instances // self.n_domains

        weight = self.weight_head(instance_node)
        weight = weight.view(self.n_domains, batch_size, -1)
        
        labels = labels.reshape((self.n_domains, batch_size))
        instance_node = instance_node.view(self.n_domains, batch_size, -1)

        agent_nodes = [] #Expected tensor with shape: (n_domains, n_classes, feat_dim)
        for domain_idx in range(labels.shape[0]):
            agent_nodes.append(self._agent_node_class_level(instance_node[domain_idx], 
                                                            weight[domain_idx],
                                                            labels[domain_idx]))
        agent_nodes = torch.stack(agent_nodes)
        # print("SHAPE OF FINAL AGENT NODE:", agent_nodes.shape)
        # print(agent_nodes[:, :, :3])
        
        return agent_nodes #shape: (n_domains, n_classes, feat_dim)

    def _agent_node_class_level(self, instance_node: torch.Tensor,
                                weight: torch.Tensor,
                                labels):
        '''
        By-class agent nodes feature for one doamin
        Input:
            instance_node: instance nodes of one domain, shape (batch_size, feat_dim)
            weight: corresponding weights (not yet normalized), shape (batch_size, feat_dim)
            labels: corresponding labels, np.array, shape (batch_size)
        '''
        agent_nodes = []

        for class_idx in range(self.n_classes):
            matched_indicies = np.where(labels == class_idx)

            #Softmax by dimension 0 (batch_size)
            w = F.softmax(weight[matched_indicies], dim= 0)
            agent_node = (w*instance_node[matched_indicies]).sum(dim = 0)

            agent_nodes.append(agent_node)

            '''
            shape of agent node here: (feat_dim)
            agent node could be zero-tensor, in case there is no instances for this class
            Return: list of agent nodes of all classes for this domain
                shape (n_classes, feat_dim)
                If the agent node is zero-tensor, do not update global agent node by regular EMA
                , must ignore this case without updating, because is is not the actual value of agent node
            '''

        return torch.stack(agent_nodes) #shape: (n_classes, feat_dim)
    
    def class_graph_training(self, instance_node: torch.Tensor,
                             labels):
        
        n_instances = instance_node.shape[0]
        batch_size = n_instances // self.n_domains
        first_kn_graph = torch.zeros(n_instances + self.n_domains*self.n_classes, 
                                     n_instances + self.n_domains*self.n_classes)
        second_kn_graph = torch.zeros(n_instances + self.n_domains*self.n_classes, 
                                     n_instances + self.n_domains*self.n_classes)
        
        for class_idx in range(self.n_classes):
            first_kn_graph[n_instances + class_idx*self.n_domains:n_instances + (class_idx+1)*self.n_domains,
                           n_instances + class_idx*self.n_domains:n_instances + (class_idx+1)*self.n_domains] = 1
            second_kn_graph[n_instances + class_idx*self.n_domains:n_instances + (class_idx+1)*self.n_domains,
                           n_instances + class_idx*self.n_domains:n_instances + (class_idx+1)*self.n_domains] = 1
            
            for domain_idx in range(self.n_domains):
                first_kn_graph[n_instances + class_idx*self.n_domains + domain_idx,
                               n_instances + class_idx*self.n_domains + domain_idx] = 0
                second_kn_graph[n_instances + class_idx*self.n_domains + domain_idx,
                               n_instances + class_idx*self.n_domains + domain_idx] = 0        

        for instance_idx in range(len(labels)):
            domain_idx = instance_idx // batch_size
            class_idx = labels[instance_idx]

            agt = self.agent_node_ema.reshape(self.n_classes, self.n_domains, -1)
            agt = agt[:, domain_idx, :]

            instance = instance_node[instance_idx]
            instance = torch.stack([instance for i in range(self.n_classes)])
            distance = torch.norm(instance - agt, p=2, dim= 1)

            #Only take the distance to non-zero agent nodes
            agt_mask = torch.any(agt != 0, dim = 1)
            distance = distance[agt_mask]
            distance_inversed = 1 / distance
            edge_weight = distance_inversed / distance_inversed.sum()

            #This edge is not necessary
            # second_kn_graph[instance_idx, n_instances + class_idx*self.n_domains + domain_idx] = 1
            for i, c_idx in enumerate(agt_mask.nonzero().squeeze(dim = 1).cpu().numpy()):
                second_kn_graph[n_instances + c_idx*self.n_domains + domain_idx, instance_idx] = edge_weight[i]
            
            #Old knowledge graph
            # second_kn_graph[instance_idx, n_instances + class_idx*self.n_domains + domain_idx] = 1
            # second_kn_graph[n_instances + class_idx*self.n_domains + domain_idx, instance_idx] = 1

        first_edge_indicies = first_kn_graph.nonzero(as_tuple=False).t()
        second_edge_indicies = second_kn_graph.nonzero(as_tuple=False).t()

        row, col = torch.nonzero(second_kn_graph, as_tuple=True)
        second_edge_weight = second_kn_graph[row, col]
        
        # print("NEW KNOWLEDGE GRAPH:")
        # print(second_kn_graph)
        # print(second_edge_weight)
        # print("NUMBER OF EDGES:", second_edge_weight.shape)

        return first_edge_indicies, second_edge_indicies, second_edge_weight
    
    def _old_class_graph_training(self, n_instances, labels):

        batch_size = n_instances // self.n_domains
        first_kn_graph = torch.zeros(n_instances + self.n_domains*self.n_classes, 
                                     n_instances + self.n_domains*self.n_classes)
        second_kn_graph = torch.zeros(n_instances + self.n_domains*self.n_classes, 
                                     n_instances + self.n_domains*self.n_classes)
        
        for class_idx in range(self.n_classes):
            first_kn_graph[n_instances + class_idx*self.n_domains:n_instances + (class_idx+1)*self.n_domains,
                           n_instances + class_idx*self.n_domains:n_instances + (class_idx+1)*self.n_domains] = 1
            second_kn_graph[n_instances + class_idx*self.n_domains:n_instances + (class_idx+1)*self.n_domains,
                           n_instances + class_idx*self.n_domains:n_instances + (class_idx+1)*self.n_domains] = 1
            
            for domain_idx in range(self.n_domains):
                first_kn_graph[n_instances + class_idx*self.n_domains + domain_idx,
                               n_instances + class_idx*self.n_domains + domain_idx] = 0
                second_kn_graph[n_instances + class_idx*self.n_domains + domain_idx,
                               n_instances + class_idx*self.n_domains + domain_idx] = 0        

        for instance_idx in range(len(labels)):
            domain_idx = instance_idx // batch_size
            class_idx = labels[instance_idx]

            #Old knowledge graph
            second_kn_graph[instance_idx, n_instances + class_idx*self.n_domains + domain_idx] = 1
            second_kn_graph[n_instances + class_idx*self.n_domains + domain_idx, instance_idx] = 1

        first_edge_indicies = first_kn_graph.nonzero(as_tuple=False).t()
        second_edge_indicies = second_kn_graph.nonzero(as_tuple=False).t()

        return first_edge_indicies, second_edge_indicies
    
    def class_graph_inference(self, instance_node: torch.Tensor, 
                              domain_idx):
        
        n_instances = instance_node.shape[0]

        batch_size = n_instances // self.n_domains
        first_kn_graph = torch.zeros(n_instances + self.n_domains*self.n_classes, 
                                     n_instances + self.n_domains*self.n_classes)
        second_kn_graph = torch.zeros(n_instances + self.n_domains*self.n_classes, 
                                     n_instances + self.n_domains*self.n_classes)
        
        for class_idx in range(self.n_classes):
            first_kn_graph[n_instances + class_idx*self.n_domains:n_instances + (class_idx+1)*self.n_domains,
                           n_instances + class_idx*self.n_domains:n_instances + (class_idx+1)*self.n_domains] = 1
            second_kn_graph[n_instances + class_idx*self.n_domains:n_instances + (class_idx+1)*self.n_domains,
                           n_instances + class_idx*self.n_domains:n_instances + (class_idx+1)*self.n_domains] = 1
            
            for d_idx in range(self.n_domains):
                first_kn_graph[n_instances + class_idx*self.n_domains + d_idx,
                               n_instances + class_idx*self.n_domains + d_idx] = 0
                second_kn_graph[n_instances + class_idx*self.n_domains + d_idx,
                               n_instances + class_idx*self.n_domains + d_idx] = 0  
                
        agt = self.agent_node_ema.reshape(self.n_classes, self.n_domains, -1)
        agt = agt[:, domain_idx, :]

        agt_mask = torch.any(agt != 0, dim = 1)
        
        for instance_idx in range(n_instances):
            instance = instance_node[instance_idx]
            instance = torch.stack([instance for i in range(self.n_classes)])
            distance = torch.norm(instance - agt, p=2, dim= 1)

            #Only take the distance to non-zero agent nodes
            distance = distance[agt_mask]
            distance_inversed = 1 / distance
            edge_weight = distance_inversed / distance_inversed.sum()

            for i, class_idx in enumerate(agt_mask.nonzero().squeeze(dim = 1).cpu().numpy()):
                #Set values by normalized distance to agent nodes
                second_kn_graph[instance_idx, n_instances + class_idx*self.n_domains + domain_idx] = edge_weight[i] 
                second_kn_graph[n_instances + class_idx*self.n_domains + domain_idx, instance_idx] = edge_weight[i]

        #Calculating distance to the agent node here
        first_edge_indicies = first_kn_graph.nonzero(as_tuple=False).t()
        second_edge_indicies = second_kn_graph.nonzero(as_tuple=False).t()

        row, col = torch.nonzero(second_kn_graph, as_tuple=True)
        second_edge_weight = second_kn_graph[row, col]

        return first_edge_indicies, second_edge_indicies, second_edge_weight

    def update_agent_node_class_level(self, agent_node_class_level: torch.Tensor):
        '''
        Input:
            agent_node_class_level: shape (n_domains, n_classes, feat_dim) 
                permute --> (n_classes, n_domains, feat_dim) to be consistant 
                with order class 1(d1, d2, d3), class 2 (d1, d2, d3)
        self.agent_node_ema: 
            shape (n_domains*n_classes, feat_dim)
            order first dimension: class 1(d1, d2, d3), class 2 (d1, d2, d3)
        '''

        # Permute agent node shape to (n_classes, n_domains, feat_dim)
        # then reshape to (-1, feat_dim), order: class 1(d1, d2, d3), class 2 (d1, d2, d3)
        agent_node_class_level = torch.permute(agent_node_class_level, (1, 0, 2))
        agent_node_class_level = agent_node_class_level.reshape(self.n_classes*self.n_domains, -1)
        
        agent_node_updated = torch.zeros_like(agent_node_class_level)
        
        for agt_idx in range(self.n_domains*self.n_classes):

            if torch.all(self.agent_node_ema[agt_idx] == 0):
                if not torch.all(agent_node_class_level[agt_idx] == 0):
                    agent_node_updated[agt_idx] = agent_node_class_level[agt_idx]

                    self.agent_node_ema[agt_idx] = agent_node_updated[agt_idx].detach()
            else:
                if not torch.all(agent_node_class_level[agt_idx] == 0):
                    agent_node_updated[agt_idx] = (1-self.ema_alpha)*self.agent_node_ema[agt_idx].detach() \
                                                    + self.ema_alpha*agent_node_class_level[agt_idx]
                    
                    self.agent_node_ema[agt_idx] = agent_node_updated[agt_idx].detach()
                else:
                    agent_node_updated[agt_idx] = self.agent_node_ema[agt_idx].detach()

        return agent_node_updated