from torch import nn
from torch_geometric.nn import GCNConv, GraphConv, global_mean_pool
from torch.functional import F
from sklearn.metrics import average_precision_score, roc_auc_score



class ConversationalGraph(nn.Module):
    def __init__(self, embedding_size, hidden_channels, num_classes):
        super(ConversationalGraph, self).__init__()
        self.gconv1 = GraphConv(embedding_size, hidden_channels)
        self.gconv2 = GraphConv(hidden_channels, hidden_channels)
        self.gconv3 = GraphConv(hidden_channels, hidden_channels)
        self.linear = GraphConv(hidden_channels, num_classes)
        self.relu = nn.LeakyReLU()

    def forward(self, x_embeddings, edge_index, weights, batch):
        x = self.gconv1(x_embeddings, edge_index, weights)
        x = self.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gconv2(x, edge_index, weights)
        x = self.relu(x)
        x = F.dropout(x, training=self.training)
        #Mean pool over graphs
        x = global_mean_pool(x, batch)
        #Graph classification
        x = F.dropout(x, p=0.5, training=self.training)
        out = self.linear(x)
        return out