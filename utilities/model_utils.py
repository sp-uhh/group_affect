import torch
from torch_geometric.data import Data
from readers.utils.utils import get_groupsize_for_batch

import constants as c

def create_adjacency_matrix(group_size, max_group_size):
    """
    Create an adjacency matrix for a given group size and max group size.

    Args:
        group_size (int): The number of nodes in the current sample's group.
        max_group_size (int, optional): The maximum number of nodes in any group. Defaults to 7.

    Returns:
        torch.Tensor: The adjacency matrix of shape [max_group_size, max_group_size].
    """
    adj_matrix = torch.zeros((max_group_size, max_group_size), dtype=torch.float)

    # Fully connect the nodes for the given group size
    for i in range(group_size):
        for j in range(group_size):
            if i != j:  # Exclude self-loops
                # if i < j:  # Exclude repeating edges
                adj_matrix[i, j] = 1


    return adj_matrix

def adj_matrix_to_edge_index(adj_matrix):
    """
    Convert an adjacency matrix to edge index format.

    Args:
        adj_matrix (torch.Tensor): The adjacency matrix of shape [max_group_size, max_group_size].

    Returns:
        torch.Tensor: The edge index of shape [2, num_edges].
    """
    edge_index = torch.nonzero(adj_matrix, as_tuple=False).t().contiguous()
    # print("Edge Index creation: ", edge_index.shape)
    return edge_index


def create_edge_index_for_batch(group_sizes, max_group_size):
    """
    Process a batch of group sizes to create edge indices for each sample.

    Args:
        group_sizes (list of int): List of group sizes for each sample in the batch.
        max_group_size (int, optional): The maximum number of nodes in any group. Defaults to 7.

    Returns:
        list of torch.Tensor: List of edge indices for each sample in the batch.
    """
    edge_indices = []
    # print("Group Sizes: ", group_sizes)
    for group_size in group_sizes:
        adj_matrix = create_adjacency_matrix(group_size, max_group_size)
        edge_index = adj_matrix_to_edge_index(adj_matrix)
        edge_indices.append(edge_index)
    return edge_indices



def setup_data_for_gcn(grp, ses, features, y_train, batch_size):
    group_sizes = get_groupsize_for_batch(grp, ses)
    edge_index = create_edge_index_for_batch(group_sizes, max_group_size=c.MAX_GROUP_SIZE)
    # create a tensor with repeating integer "n" number of times ranging from 0-batch_size
    batch = torch.arange(batch_size).repeat_interleave(c.MAX_GROUP_SIZE)
    data = Data(x=features, edge_index=edge_index, y=y_train, batch=batch)
    return data