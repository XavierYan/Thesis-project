import torch
import torch.nn as nn
import torch.nn.functional as F  # useful stateless functions
# import dgl
# import networkx as nx
import numpy as np
# import matplotlib.animation as animation
import matplotlib.pyplot as plt
import warnings



class GraphAttentionNet(nn.Module):
    def __init__(self, in_dim=256, key_dim=128, value_dim=128):
        super().__init__()
        self.queryFC = nn.Linear(in_dim, key_dim)
        nn.init.kaiming_normal_(self.queryFC.weight)
        self.keyFC = nn.Linear(in_dim, key_dim)
        nn.init.kaiming_normal_(self.keyFC.weight)
        self.valueFC = nn.Linear(in_dim, value_dim)
        nn.init.kaiming_normal_(self.valueFC.weight)

    def forward(self, polyline_feature):
        # [batch_size, N, in_dim]
        p_query = F.relu(self.queryFC(polyline_feature))  # [batch_size, N, key_dim]
        p_key = F.relu(self.keyFC(polyline_feature))  # [batch_size, N, key_dim]
        p_value = F.relu(self.valueFC(polyline_feature))  # [batch_size, N, value_dim]

        query_result = torch.bmm(p_query, p_key.transpose(1, 2))  # [batch_size, N, N]

        query_result = query_result / (p_key.shape[2]** 0.5)

        attention = F.softmax(query_result, dim=-1)  # [batch_size, N, N]


        output = torch.bmm(attention, p_value)  # [batch_size, N, value_dim]


        return output + p_query
