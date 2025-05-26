import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphSAGE


# Define the Graph Neural Network model
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x.view(-1)


# Define GAT Model for Binary Classification
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=8):
        super(GAT, self).__init__()
        # First GAT layer (multi-head attention)
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        # Second GAT layer (single-head for binary output)
        self.conv2 = GATConv(
            hidden_channels * heads, 1, heads=1, concat=False, dropout=0.6
        )

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)  # Use ELU activation function
        x = self.conv2(x, edge_index)
        return x.view(-1)  # Output raw logits (no sigmoid here)
