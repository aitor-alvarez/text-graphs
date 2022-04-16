import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import Linear
from data.dataloader import TemporalDataLoader
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)

class TimeConversationalGraph():

