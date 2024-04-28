from torch import nn
from torch.nn import functional as F
import dgl
from dgl.nn.pytorch.conv import RelGraphConv
from dgl.nn.pytorch import SumPooling
from dgllife.model.gnn import WLN


class FCLayer(nn.Module):
    def __init__(self, dropout, in_feats, hidden_feats):
        super(FCLayer, self).__init__()

        self.fc_layer = nn.Sequential(
            #nn.Dropout(dropout),
            nn.Linear(in_feats, hidden_feats),
            nn.ReLU()
        )
    
    def forward(self, feats):
        return self.fc_layer(feats)

class RGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_rels=65, activation=F.relu, self_loop=False, dropout=0.5):
        super(RGCNLayer, self).__init__()
        
        self.graph_conv_layer = RelGraphConv(in_feats, out_feats, num_rels=num_rels, regularizer='basis',
                                             num_bases=None, bias=True, activation=activation,
                                             self_loop=self_loop, dropout=dropout)
    
    def forward(self, g, node_feats, edge_feats):
        return self.graph_conv_layer(g, node_feats, edge_feats)
    
class RGCN(nn.Module):
    def __init__(self, node_size, hidden_size, gnn_layers=3, fc_layers=3, rgcn_dropout=0.25, fc_dropout=0.25):
        super(RGCN, self).__init__()

        self.gnn_layers = [RGCNLayer(node_size, hidden_size, dropout=rgcn_dropout)]
        for _ in range(gnn_layers - 1):
            self.gnn_layers.append(RGCNLayer(hidden_size, hidden_size, dropout=rgcn_dropout))

        self.fc_layers = []
        for _ in range(fc_layers):
            self.fc_layers.append(FCLayer(fc_dropout, hidden_size, hidden_size))

        self.predict = nn.Linear(hidden_size, 2)

    def forward(self, g, node_feats, edge_feats, embed=False):
        # Update atom features with GNNs
        for layer in self.gnn_layers:
            node_feats = layer(g, node_feats, edge_feats)

        if embed:
            return node_feats

        # Compute molecule features from atom features and bond features
        graph_feats = self.sum_nodes(g, node_feats)

        for layer in self.fc_layers:
            graph_feats = layer(graph_feats)

        out = self.predict(graph_feats)

        return out
    
    def sum_nodes(self, g, feats):
        with g.local_scope():
            g.ndata['h'] = feats
            h_g_sum = dgl.sum_nodes(g, 'h')

            return h_g_sum
    
class WLNClassifier(nn.Module):
    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 num_classes,
                 node_hidden_feats=300,
                 num_encode_gnn_layers=3):
        super(WLNClassifier, self).__init__()

        self.gnn = WLN(node_in_feats=node_in_feats,
                       edge_in_feats=edge_in_feats,
                       node_out_feats=node_hidden_feats,
                       n_layers=num_encode_gnn_layers,
                       set_comparison=False)
        
        self.readout = SumPooling()

        self.predict = nn.Sequential(
            nn.Linear(node_hidden_feats, node_hidden_feats),
            nn.ReLU(),
            nn.Linear(node_hidden_feats, num_classes)
        )

    def forward(self, graph, node_feats, edge_feats, embed=False, mask=None):
        node_feats = self.gnn(graph, node_feats, edge_feats)

        if embed:
            return node_feats
        
        if mask is not None:
            node_feats = node_feats * mask[:, None]

        graph_feats = self.readout(graph, node_feats)

        return self.predict(graph_feats)

