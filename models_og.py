import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn import GraphConv, GATConv, GINConv, MaxPooling, AvgPooling, GlobalAttentionPooling
from dgllife.model.gnn.wln import WLN
from dgllife.model.gnn import GAT

class GCNLayer(nn.Module):
    """ Simplified GCN layer with residual, dropout and batchnorm
    """
    def __init__(self, in_feats, out_feats, dropout=0.1, activation=None):
        super(GCNLayer, self).__init__()
        
        self.activation = activation
        self.gnn = GraphConv(in_feats, out_feats, norm="both", activation=None)
        self.dropout = nn.Dropout(dropout)
        self.res_connection = nn.Linear(in_feats, out_feats)
        self.bn = nn.BatchNorm1d(out_feats)
        
    def reset_parameters(self):
        self.gnn.reset_parameters()
        self.res_connection.reset_parameters()
        self.bn.reset_parameters()
        
    def forward(self, g, feats):
        new_feats = self.gnn(g, feats)
        res_feats = self.res_connection(feats)
        if self.activation is not None:
            res_feats = F.relu(res_feats)
        return self.bn(self.dropout(new_feats + res_feats))
        
        

class GATLayer(nn.Module):
    """Simplified GAT layer with residual, dropout and batchnorm
    """
    def __init__(self, in_feats, out_feats, num_heads, dropout=0.1, 
                 activation=None):
        super(GATLayer, self).__init__()
        
        self.activation = activation
        self.gnn = GATConv(
            in_feats, out_feats, num_heads, feat_drop=dropout, residual=True,
            activation = activation
        )
        self.bn = nn.BatchNorm1d(out_feats)
        
    def reset_parameters(self):
        self.gnn.reset_parameters()
        self.bn.reset_parameters()
        
    def forward(self, g, feats):
        new_feats = self.gnn(g, feats)
        return self.bn(new_feats.mean(1))

class GINNodeFunc(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GINNodeFunc, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_feats, out_feats))
        self.layers.append(nn.BatchNorm1d((out_feats)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(out_feats, out_feats))
        self.layers.append(nn.BatchNorm1d((out_feats)))
        self.layers.append(nn.ReLU())
        
    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x
        
class GINLayer(nn.Module):
    """Simplified GIN layer with dropout and batchnorm
    """
    def __init__(self, in_feats, out_feats, gin_agg='max', dropout=0.1):
        super(GINLayer, self).__init__()
        mlp = GINNodeFunc(in_feats, out_feats)
        self.gnn = GINConv(mlp, gin_agg, 0, False)
        self.bn = nn.BatchNorm1d(out_feats)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, g, feats):
        return self.dropout(self.bn(self.gnn(g, feats)))

class LinearBlock(nn.Module):
    def __init__(self, in_feats, out_feats, dropout=0.1):
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.bn = nn.BatchNorm1d(out_feats)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.bn(self.dropout(F.relu(self.linear(x))))
    

class MSGNN(nn.Module):
    def __init__(self, gnn_type, num_gnn_layers, num_node_feat, num_edge_feat, 
                 gnn_out_feat, global_pooling, instrument_on_graph, 
                 num_mlp_layers, mlp_out_feat, instrument_setting_size,
                 glu, n_pred = 1000, gat_num_heads=4, gin_agg='max', 
                 dropout=0.1, activation=F.relu, virtual_node = False
                ):
        super(MSGNN, self).__init__()
        
        self.gnn_type = gnn_type
        self.gnn = nn.ModuleList()
        if gnn_type == "gcn":
            self.gnn.append(GCNLayer(
                num_node_feat, gnn_out_feat, dropout, activation))
            for i in range(num_gnn_layers - 1):
                self.gnn.append(GCNLayer(gnn_out_feat, gnn_out_feat, 
                                         dropout, activation))
        elif gnn_type == "gat":
            
            self.gnn.append(GAT(num_node_feat, [gnn_out_feat for _ in range(num_gnn_layers)],
                               [gat_num_heads for _ in range(num_gnn_layers)], 
                                feat_drops = [dropout for _ in range(num_gnn_layers)],
                               activations = [F.elu for _ in range(num_gnn_layers)]))
#             self.gnn.append(GATLayer(
#                 num_node_feat, gnn_out_feat, gat_num_heads, dropout, activation))
#             for i in range(num_gnn_layers - 1):
#                 self.gnn.append(GATLayer(
#                     gnn_out_feat, gnn_out_feat, gat_num_heads, 
#                     dropout, activation))
        elif gnn_type == "gin":
            self.gnn.append(GINLayer(
                num_node_feat, gnn_out_feat, gin_agg, dropout))
            for i in range(num_gnn_layers - 1):
                self.gnn.append(GINLayer(
                    gnn_out_feat, gnn_out_feat, gin_agg, dropout))
        elif gnn_type == "wln":
            self.gnn.append(WLN(num_node_feat, num_edge_feat, 
                           gnn_out_feat, num_gnn_layers))
                
        if global_pooling == "max":
            self.pool = MaxPooling()
        elif global_pooling == "avg":
            self.pool = AvgPooling()
        elif global_pooling == "attn":
            pooling_gate_nn = nn.Linear(gnn_out_feat, 1)
            self.pool = GlobalAttentionPooling(pooling_gate_nn)
            
        mlp_list = []
        self.instrument_on_graph = instrument_on_graph
        if instrument_on_graph:
            mlp_list.append(LinearBlock(
                gnn_out_feat + instrument_setting_size, mlp_out_feat, dropout
            ))
        else:
            mlp_list.append(LinearBlock(gnn_out_feat, mlp_out_feat, dropout))
        for i in range(num_mlp_layers - 1):
            mlp_list.append(LinearBlock(mlp_out_feat, mlp_out_feat, dropout))
        self.mlp = nn.Sequential(*mlp_list)
        
        self.glu = glu
        if glu:
            self.final = nn.Linear(mlp_out_feat, n_pred * 2)
        else:
            self.final = nn.Linear(mlp_out_feat, n_pred)
        
        self.virtual_node = virtual_node
            
    def graphembedding(self, g, f, ef):
        if self.gnn_type in ['gcn', 'gat', 'gin']:
            for layer in self.gnn:
                f = layer(g, f)
        else:
            for layer in self.gnn:
                f = layer(g, f, ef)
        if self.virtual_node:
            h = f.narrow(0, g.num_nodes() - 1, 1)
        else:
            h = self.pool(g, f)
        return F.relu(h)
    
    def forward(self, g, f, ef, s):
        h = self.graphembedding(g, f, ef)
        
        if self.instrument_on_graph:
            h = torch.cat((h, s), dim = 1)
        
        h = self.final(self.mlp(h))
        
        if self.glu:
            h = F.glu(h)
            
        return F.relu(h)
    