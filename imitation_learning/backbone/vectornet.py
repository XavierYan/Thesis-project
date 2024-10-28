import torch
import torch.nn as nn
import torch.nn.functional as F  # useful stateless functions
import numpy as np
# import matplotlib.animation as animation
import matplotlib.pyplot as plt
from .subgraph_net import SubgraphNet, SubgraphNet4state
from .gnn import GraphAttentionNet
import sys

import warnings
warnings.filterwarnings('ignore')


class VectorNet(nn.Module):
    def __init__(self, state=7, reference_traj = 5, map_features=5, cfg=None):
    # def __init__(self, state=9, reference_traj = 5, map_features=8, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = dict(device=torch.device('cuda:0'))
        self.cfg = cfg
        self.traj_subgraphnet = SubgraphNet4state(state)
        self.reference_subgraphnet = SubgraphNet(reference_traj)
        self.map_subgraphnet = SubgraphNet(map_features)
        self.graphnet = GraphAttentionNet()
        # decoder
        prediction_step = 50  
        self.fc = nn.Linear(128, 128)
        nn.init.kaiming_normal_(self.fc.weight)
        self.layer_norm = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 64)
        nn.init.kaiming_normal_(self.fc2.weight)
        self.fc3 = nn.Linear(512,256)
        self.layer_norm3 = nn.LayerNorm(256)
        nn.init.kaiming_normal_(self.fc3.weight)
        self.fc4 = nn.Linear(128,128)
        nn.init.kaiming_normal_(self.fc4.weight)

        self.loss_fn = nn.MSELoss(reduction='sum') 




    def forward(self, ego_trajectory_batch, ego_reference_batch, neighbour_trajectory_batch, neighbour_reference_batch, vectormap_batch):
        # 直接处理整个批次的ego_trajectory和ego_reference
        ego_traj_features = self.traj_subgraphnet(ego_trajectory_batch)  # [batch_size, feature_dim]
        ego_ref_features = self.reference_subgraphnet(ego_reference_batch)  # [batch_size,  feature_dim]

        # concat ego feature
        ego_features = torch.cat([ego_traj_features, ego_ref_features], dim=1)  # [batch_size,  feature_dim*2]


        # [batch_size, num_neighbours, N，feature_dim]
        neighbour_traj_features = self.traj_subgraphnet(neighbour_trajectory_batch.reshape(-1, *neighbour_trajectory_batch.shape[2:])).reshape(*neighbour_trajectory_batch.shape[:2], -1)
        neighbour_ref_features = self.reference_subgraphnet(neighbour_reference_batch.reshape(-1, *neighbour_reference_batch.shape[2:])).reshape(*neighbour_reference_batch.shape[:2], -1)

        # concat neighbors feature
        neighbour_features = torch.cat([neighbour_traj_features, neighbour_ref_features], dim=2)  # [batch_size, num_neighbours, feature_dim*2]

        ego_neighbour_features = torch.cat((ego_features.unsqueeze(1),neighbour_features),dim=1)
        ego_neighbour_features = F.relu(self.layer_norm3(self.fc3(ego_neighbour_features)))

        # process map infomation
        map_features = self.map_subgraphnet(vectormap_batch.reshape(-1, *vectormap_batch.shape[2:])).reshape(*vectormap_batch.shape[:2], -1)  # [batch_size, num_maps,  feature_dim]


        combined_features = torch.cat([ego_neighbour_features, map_features], dim=1)  # [batch_size, num_features,  feature_dim]

        #attention layer
        attention_output = self.graphnet(combined_features)  # [batch_size, num_features, 64]


        hidden = F.relu(self.layer_norm(self.fc4(attention_output[:, 0, :]))) #just use first feature to predict
        hidden = F.relu(self.layer_norm(self.fc(hidden)))
        predictions = self.fc2(hidden)

        return predictions
    
    


class MultiAgentVectorNetActor(nn.Module):
    def __init__(self, n_agents = 4, network = None,  prediction_step = 4, centralised=False, share_params=True, cfg=None):
        super().__init__()
        self.n_agents = n_agents
        self.centralised = centralised
        self.share_params = share_params
        self.prediction_step = prediction_step

        if cfg is None:
            cfg = dict(device=torch.device('cuda:0'))
        self.cfg = cfg

        if self.share_params:
            self.vector_net = network

        self.fc1 = nn.Linear(64, 64)
        nn.init.kaiming_normal_(self.fc1.weight)

        self.fc2 = nn.Linear(64, 4)
        # self.fc3 = nn.Linear(5, 64)
        # self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)

        self.loss_fn = nn.MSELoss(reduction='sum')


    def forward(self, ego_trajectory_batch, ego_reference_batch, neighbour_trajectory_batch, neighbour_reference_batch, vectormap_batch):

        output = self.vector_net(
            ego_trajectory_batch,
            ego_reference_batch,
            neighbour_trajectory_batch,
            neighbour_reference_batch,
            vectormap_batch
        )
        output = F.relu(self.fc1(output))
        output = self.fc2(output)
        
        # Modify the output using non-in-place operations
        modified_output = torch.cat([
            torch.tanh(output[:, 0:1]) * 14,             # Scale and modify first element
            torch.tanh(output[:, 1:2]) * (torch.pi/6),  # Scale and modify second element
            output[:, 2:]                   # Apply softplus to the rest
        ], dim=1)
        
        return modified_output
    

    def custom_loss(self,output, target):
        target[:,1::2] = target[:,1::2] *torch.pi/180
        mean = output[:, :2]  
        log_var = output[:, 2:]  
        var = torch.exp(log_var)  


        nll_loss = 0.5 * torch.sum(log_var + (target[:, :2] - mean) ** 2 / var)
        return nll_loss.mean() 


