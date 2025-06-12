import torch
from custom_layers import global_var_pool
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.nn import global_mean_pool, GCNConv
import torch.nn.functional as F


class Normalizer(torch.nn.Module):
    def __init__(self, mean, std):
        super(Normalizer, self).__init__()
        self.mean = torch.nn.Parameter(mean, requires_grad=False)
        self.std = torch.nn.Parameter(std, requires_grad=False)

    def forward(self, x):
        return (x - self.mean) / self.std
    
    def estimate(self, dataset):
        """Estimate mean and std from the dataset."""
        all_features = dataset.x
        self.mean.data = all_features.mean(dim=0)
        self.std.data = all_features.std(dim=0)
        self.std.data[self.std.data == 0] = 1.0  # Avoid division by zero
        
        
class OutputNormalizer(torch.nn.Module):
    def __init__(self, mean, std, is_classification):
        super(OutputNormalizer, self).__init__()
        self.mean = torch.nn.Parameter(mean, requires_grad=False)
        self.std = torch.nn.Parameter(std, requires_grad=False)
        self.is_classification = is_classification

    def forward(self, x):
        if self.is_classification:
            return x
        return (x - self.mean) / self.std
    
    def unnormalize(self, x):
        if self.is_classification:
            return x
        return x * self.std + self.mean
    
    def estimate(self, dataset):
        """Estimate mean and std from the dataset."""
        if self.is_classification:
            return # No normalization for classification tasks
        all_outputs = dataset.target
        self.mean.data = all_outputs.mean()
        self.std.data = all_outputs.std()
        self.std.data[self.std.data == 0] = 1.0  # Avoid division by zero

class SimpleGraphConv(torch.nn.Module):
    """Simple graph convolution for graph classification

    Keyword Arguments
    -----------------
        node_feature_dim : Dimension of the node features
        filter_length : Length of convolution filter
    """

    def __init__(self, node_feature_dim, filter_length, normalizer, classification=False):
        super().__init__()

        # Define dimensions and other hyperparameters
        self.node_feature_dim = node_feature_dim
        self.filter_length = filter_length

        self.normalizer = normalizer
        # Define graph filter
        self.h = torch.nn.Parameter(1e-5*torch.randn(filter_length))
        self.h.data[0] = 1.

        # State output network
        self.output_net = torch.nn.Linear(self.node_feature_dim, 1)
        self.classification = classification

        self.cached = False

    def forward(self, x, edge_index, batch):
        """Evaluate neural network on a batch of graphs.

        Parameters
        ----------
        x : torch.tensor (num_nodes x num_features)
            Node features.
        edge_index : torch.tensor (2 x num_edges)
            Edges (to-node, from-node) in all graphs.
        batch : torch.tensor (num_nodes)
            Index of which graph each node belongs to.

        Returns
        -------
        out : torch tensor (num_graphs)
            Neural network output for each graph.

        """
        # Extract number of nodes and graphs
        num_graphs = batch.max()+1

        # Compute adjacency matrices and node features per graph
        A = to_dense_adj(edge_index, batch)
        # Normalize the node features
        #x = self.normalizer(x)
        X, idx = to_dense_batch(x, batch)
        
        # Implementation in vertex domain
        node_state = torch.zeros_like(X)
        for k in range(self.filter_length):
            node_state += self.h[k] * torch.linalg.matrix_power(A, k) @ X

        # Aggregate the node states
        graph_state = node_state.sum(1)

        # Output
        out = self.output_net(graph_state).flatten()
        if self.classification:
            out = torch.sigmoid(out)
        return out
    
class GCNConv2(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.bias = torch.nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x)

        # Step 6: Apply a final bias vector.
        out = out + self.bias

        return out

    def message(self, x_j):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return x_j
    
    
class MessagePassingGNN(torch.nn.Module):
    def __init__(self, in_channels, filter_size, n_rounds, normalizer, is_classification=False, dropout=0.5):
        """Message Passing GNN for graph classification.
        Parameters
        ----------
        in_channels : int
            Number of input features per node.
        filter_size : int
            Number of output features per node after each convolution.
        n_rounds : int
            Number of rounds of message passing.
        normalizer : Normalizer
            Normalizer for node features.
        is_classification : bool
            Whether the task is classification (default: False).
        dropout : float
            Dropout rate (default: 0.5).
        """
        super(MessagePassingGNN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, filter_size, improved=True))
        for i in range(n_rounds):
            self.convs.append(GCNConv(filter_size, filter_size, improved=True))
        self.fc1 = torch.nn.Linear(filter_size, filter_size)
        self.fc2 = torch.nn.Linear(filter_size, 1)
        self.is_classification = is_classification
        self.normalizer = normalizer
        self.dropout_rate = dropout

    def forward(self, x, edge_index, batch, batch_size, return_var=False):
        """Evaluate neural network on a batch of graphs.
        Parameters
        ----------
        x : torch.tensor (num_nodes x num_features)
            Node features.
        edge_index : torch.tensor (2 x num_edges)
            Edges (to-node, from-node) in all graphs.
        batch : torch.tensor (num_nodes)
            Index of which graph each node belongs to.
        Returns
        -------
        out : torch tensor (num_graphs)
            Neural network output for each graph.
        """
        # Normalize the node features
        #x = self.normalizer(x)
        
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            
        x = global_mean_pool(x, batch, size=batch_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        if self.is_classification:
            x = torch.sigmoid(x)
        x = x[:, 0]
        if return_var:
            return x, x*0
        return x
        


class EnsembleConnectedGNN(torch.nn.Module):
    def __init__(self, in_channels, filter_size, n_rounds, normalizer, is_classification=False, dropout=0.5):
        """Message Passing GNN for graph classification.
        Parameters
        ----------
        in_channels : int
            Number of input features per node.
        filter_size : int
            Number of output features per node after each convolution.
        n_rounds : int
            Number of rounds of message passing.
        normalizer : Normalizer
            Normalizer for node features.
        is_classification : bool
            Whether the task is classification (default: False).
        dropout : float
            Dropout rate (default: 0.5).
        """
        super(EnsembleConnectedGNN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.linear_probes = torch.nn.ModuleList()
        self.linear_probes.append(torch.nn.Linear(filter_size, 1))
        self.convs.append(GCNConv(in_channels, filter_size, improved=True))
        for i in range(n_rounds):
            self.convs.append(GCNConv(filter_size, filter_size, improved=True))
            self.linear_probes.append(torch.nn.Linear(filter_size, 1))
        self.fc = torch.nn.Linear(filter_size, 1)
        self.h = torch.nn.Parameter(1e-5*torch.randn(n_rounds + 1))
        self.h.data[-1] = 1.
        self.is_classification = is_classification
        self.normalizer = normalizer
        self.dropout_rate = dropout

    def forward(self, x, edge_index, batch, batch_size, return_var=False):
        """Evaluate neural network on a batch of graphs.
        Parameters
        ----------
        x : torch.tensor (num_nodes x num_features)
            Node features.
        edge_index : torch.tensor (2 x num_edges)
            Edges (to-node, from-node) in all graphs.
        batch : torch.tensor (num_nodes)
            Index of which graph each node belongs to.
        Returns
        -------
        out : torch tensor (num_graphs)
            Neural network output for each graph.
        """
        # Normalize the node features
        #x = self.normalizer(x)
        
        means = []
        vars = []
        
        for conv, linear in zip(self.convs, self.linear_probes):
            # Message passing
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            
            # Linear probe
            linear_out = linear(x)
            linear_mean = global_mean_pool(linear_out, batch, size=batch_size)
            linear_var = global_var_pool(linear_out, batch, size=batch_size)
            means.append(linear_mean)
            vars.append(linear_var)
            
        # Concatenate means and vars
        x = torch.stack(means, dim=1)
        x = (torch.softmax(self.h, dim=-1)[None, :, None] * x).sum(dim=1)
        vars = torch.stack(vars, dim=1)
        vars = (torch.softmax(self.h, dim=-1)[None, :, None] * vars).sum(dim=1)[:, 0]
            
        if self.is_classification:
            x = torch.sigmoid(x)
        x = x[:, 0]
        if return_var:
            return x, vars
        return x