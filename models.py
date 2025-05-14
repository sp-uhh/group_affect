import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

from torch_geometric.nn import GATConv, GraphConv
from torch_geometric.data import Data as gData
from torch_geometric.data import Batch
import constants as c

cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MLP(nn.Module):
    
    def __init__(self, config):
        super(MLP, self).__init__()
        
        self.lstm = nn.LSTM(config.nfeatures, config.nfeatures,
                            2, bidirectional=False, batch_first=True)

        self.dense1 = nn.Linear(config.nfeatures, 128)
        self.act1   = nn.ReLU(inplace=True)
        self.norm1  = nn.BatchNorm1d(128, affine=True)
        
        self.dense2 = nn.Linear(128, 64)
        self.act2    = nn.ReLU(inplace=True)
        self.norm2  = nn.BatchNorm1d(64, affine=True)
        
        self.out = nn.Linear(64, config.nlabels)
        
    def forward(self, x):
        
        x, (hn, cn)  = self.lstm(x)
                
        x = self.dense1(x)
        x = self.act1(x)
        x = self.norm1(x)
        
        x = self.dense2(x)
        x = self.act2(x)
        x = self.norm2(x)
        
        x = self.out(x)
        
        return x
    
    
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, batch_size=16, dropout=0.6):
        super(GAT, self).__init__()
        print("Building GAT Model")
        print("In Channels: ", in_channels)
        print("Hidden Channels: ", hidden_channels)
        print("Out Channels: ", out_channels)
        print("Heads: ", heads)
        self.batch_size = batch_size

        self.conv1 = GATConv(in_channels, 128, heads=heads)
        self.act1  = nn.ReLU(inplace=True)
        self.conv2 = GATConv(128 * heads, 256, heads=heads)
        self.act2  = nn.ReLU(inplace=True)
        self.conv3 = GATConv(256 * heads, 128, heads=heads)
        self.act3  = nn.ReLU(inplace=True)

        self.lstm = nn.LSTM(128 * heads, 128, 2, bidirectional=True, batch_first=True)

        self.fc1 = torch.nn.Linear((128 * 2), 256)
        self.norm1  = nn.BatchNorm1d(256, affine=True)
        self.act4 = nn.ReLU(inplace=True)
        self.fc2 = torch.nn.Linear(256, 64)
        self.norm2  = nn.BatchNorm1d(64, affine=True)
        self.act5 = nn.ReLU(inplace=True)
        self.fc3 = torch.nn.Linear(64, out_channels)

        self.dropout = dropout

    
    def build_graph_batch(self, x_list,  edge_index_list):
        data_list = []
        for graph_nodes, ei in zip(x_list, edge_index_list):
            data_list.append(gData(x=graph_nodes, edge_index=ei))
        graph = Batch.from_data_list(data_list)
        return graph
    
    def forward(self, x, edge_index):

        # Shape of x: [batch_size, num_nodes, num_node_features]
        batch = torch.arange(self.batch_size).repeat_interleave(c.MAX_GROUP_SIZE) #.to(cuda)
        
        if x.shape[0] != c.MAX_GROUP_SIZE: # i.e., batch size = 1 (currently only in explainer)
            # Do for training and validation
            graph = self.build_graph_batch(x, edge_index)
            x = graph.x
            edge_index = graph.edge_index

        # GAT 1
        x, (edges1, attn1_wgt) = self.conv1(x, edge_index=edge_index, return_attention_weights=True)
        x = self.act1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # GAT 2
        x, (edges2, attn2_wgt) = self.conv2(x, edge_index=edge_index, return_attention_weights=True)
        x = self.act2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # GAT 3
        x, (edges3, attn3_wgt) = self.conv3(x, edge_index=edge_index, return_attention_weights=True)
        x = self.act3(x)

        # LSTM
        x, (hn, cn)  = self.lstm(x)

        x = global_mean_pool(x, batch)  # Shape of x: [batch_size*num_nodes, hidden_channels * heads]
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc1(x) 
        x = self.act4(x)
        x = self.norm1(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = self.act5(x)
        x = self.norm2(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)  # Shape of x: [batch_size, out_channels]

        return x , (attn1_wgt, attn2_wgt, attn3_wgt, edges1)
