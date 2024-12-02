import torch.nn as nn
# from torch_geometric.nn import GCNConv
import torch.nn.functional as F

from mmpretrain.registry import MODELS


@MODELS.register_module()
class MultiDomainInformationFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MultiDomainInformationFusion, self).__init__()
        # self.conv1 = GCNConv(input_dim, hidden_dim)  # First GCN layer
        # self.conv2 = GCNConv(hidden_dim, output_dim)  # Second GCN layer

        print("AAAAAAAAAAAAAAAAAAAAAAA")
    def forward(self, x, edge_index):
        """
        Args:
            x: Node feature matrix (num_nodes x input_dim)
            edge_index: Graph connectivity (2 x num_edges)
                        Each column represents an edge (source, target).
        """
        x = self.conv1(x, edge_index)  # Apply first GCN layer
        x = F.relu(x)                 # Non-linear activation
        x = self.conv2(x, edge_index)  # Apply second GCN layer
        print("SHAPE OF OUTPUT OF GCN MULTI DOMAIN FUSION:", x.shape)
        return x
