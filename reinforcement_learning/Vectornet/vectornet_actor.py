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
    def __init__(self, state=7, reference_traj=5, map_features=5, cfg=None):
        super(VectorNet, self).__init__()
        if cfg is None:
            cfg = dict(device=torch.device('cuda:0'))
        self.cfg = cfg
        self.device = cfg['device']
        
        self.traj_subgraphnet = SubgraphNet4state(state)
        self.reference_subgraphnet = SubgraphNet(reference_traj)
        self.map_subgraphnet = SubgraphNet(map_features)
        self.graphnet = GraphAttentionNet()
        
        self.fc = nn.Linear(128, 128)
        self.layer_norm = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(512, 256)
        self.layer_norm3 = nn.LayerNorm(256)
        self.fc4 = nn.Linear(128, 128)

        self.loss_fn = nn.MSELoss(reduction='sum')



    def forward(self, ego_trajectory_batch, ego_reference_batch, neighbour_trajectory_batch, neighbour_reference_batch, vectormap_batch):
        
        # print(f"ego_trajectory_batch device: {ego_trajectory_batch.device}")
        # print(f"ego_reference_batch device: {ego_reference_batch.device}")
        # print(f"neighbour_trajectory_batch device: {neighbour_trajectory_batch.device}")
        # print(f"neighbour_reference_batch device: {neighbour_reference_batch.device}")
        # print(f"vectormap_batch device: {vectormap_batch.device}")

        ego_traj_features = self.traj_subgraphnet(ego_trajectory_batch)  # [batch_size, feature_dim]
        ego_ref_features = self.reference_subgraphnet(ego_reference_batch)  # [batch_size,  feature_dim]

        # concat ego feature
        ego_features = torch.cat([ego_traj_features, ego_ref_features], dim=1)  # [batch_size,  feature_dim*2]


        #  [batch_size, num_neighbours, N，feature_dim]
        neighbour_traj_features = self.traj_subgraphnet(neighbour_trajectory_batch.reshape(-1, *neighbour_trajectory_batch.shape[2:])).reshape(*neighbour_trajectory_batch.shape[:2], -1)
        neighbour_ref_features = self.reference_subgraphnet(neighbour_reference_batch.reshape(-1, *neighbour_reference_batch.shape[2:])).reshape(*neighbour_reference_batch.shape[:2], -1)

        # concat neighbors' feature
        neighbour_features = torch.cat([neighbour_traj_features, neighbour_ref_features], dim=2)  # [batch_size, num_neighbours, feature_dim*2]

        ego_neighbour_features = torch.cat((ego_features.unsqueeze(1),neighbour_features),dim=1)
        ego_neighbour_features = F.relu(self.layer_norm3(self.fc3(ego_neighbour_features)))

        # process map info
        map_features = self.map_subgraphnet(vectormap_batch.reshape(-1, *vectormap_batch.shape[2:])).reshape(*vectormap_batch.shape[:2], -1)  # [batch_size, num_maps,  feature_dim]


        combined_features = torch.cat([ego_neighbour_features, map_features], dim=1)  # [batch_size, num_features,  feature_dim]

        # self-attention
        attention_output = self.graphnet(combined_features)  # [batch_size, num_features, 64]

        #decoder layer
        hidden = F.relu(self.layer_norm(self.fc4(attention_output[:, 0, :]))) # just use first feature to predict
        hidden = F.relu(self.layer_norm(self.fc(hidden)))
        hidden = self.fc2(hidden)
        # hidden = F.relu(self.fc2(hidden))
        # hidden = self.fc5(hidden)

        return hidden
    

class MultiAgentVectorNetActor(nn.Module):
    def __init__(self, n_agents: int, network,  prediction_step = 4, centralised=False, share_params=True, cfg=None):
        super().__init__()
        self.n_agents = n_agents
        self.centralised = centralised
        self.share_params = share_params

        if cfg is None:
            cfg = dict(device=torch.device('cuda:0'))
        self.cfg = cfg

        if self.share_params:
            self.vector_net = network
        # else:

        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, prediction_step)

    def forward(self, ego_trajectory_batch, ego_reference_batch, neighbour_trajectory_batch, neighbour_reference_batch, vectormap_batch):
        if self.share_params:
            outputs = []
            for i in range(self.n_agents):

                output = self.vector_net(
                    ego_trajectory_batch[:, i, :],
                    ego_reference_batch[:, i, :],
                    neighbour_trajectory_batch[:, i, :, :],
                    neighbour_reference_batch[:, i, :, :],
                    vectormap_batch[:, i, :]
                )
                output = F.relu(self.fc1(output))
                output = self.fc2(output)
                outputs.append(output)
            # print("vector", outputs)
            return torch.stack(outputs, dim=1)
        # else:
        #     outputs = []
        #     for i in range(self.n_agents):
        #         output = self.vector_nets[i](
        #             ego_trajectory_batch[:, i, :],
        #             ego_reference_batch[:, i, :],
        #             neighbour_trajectory_batch[:, i, :, :],
        #             neighbour_reference_batch[:, i, :, :],
        #             vectormap_batch[:, i, :]
        #         )
        #         outputs.append(output)
        #     return torch.stack(outputs, dim=1)

class MultiAgentVectorNetCritic(nn.Module):
    def __init__(self, n_agents: int, network,  prediction_step = 1, centralised=True, share_params=True, cfg=None):
        super().__init__()
        self.n_agents = n_agents
        self.centralised = centralised
        self.share_params = share_params

        if cfg is None:
            cfg = dict(device=torch.device('cuda:0'))
        self.cfg = cfg

        if centralised:
            # if centralised, concat all feature
            input_size = 64 * n_agents 
        else:
            input_size = 64  

        if self.share_params:
            self.vector_net = network
        # else:
        #     self.vector_nets = nn.ModuleList([VectorNet(state, reference_traj, map_features, cfg) for _ in range(n_agents)])
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, prediction_step)

    def forward(self, ego_trajectory_batch, ego_reference_batch, neighbour_trajectory_batch, neighbour_reference_batch, vectormap_batch):
        if self.share_params:

            outputs = []

            if ego_trajectory_batch.dim() == 5:
                ego_trajectory_batch = ego_trajectory_batch.view(-1, *ego_trajectory_batch.shape[2:])
                ego_reference_batch = ego_reference_batch.view(-1, *ego_reference_batch.shape[2:])
                neighbour_trajectory_batch = neighbour_trajectory_batch.view(-1, *neighbour_trajectory_batch.shape[2:])
                neighbour_reference_batch = neighbour_reference_batch.view(-1, *neighbour_reference_batch.shape[2:])
                vectormap_batch = vectormap_batch.view(-1, *vectormap_batch.shape[2:])
                for i in range(self.n_agents):

                    output = self.vector_net(
                        ego_trajectory_batch[:, i, :],
                        ego_reference_batch[:, i, :],
                        neighbour_trajectory_batch[:, i, :, :],
                        neighbour_reference_batch[:, i, :, :],
                        vectormap_batch[:, i, :]
                    )
                    outputs.append(output)
                if self.centralised:
                    combined_features = torch.cat(outputs, dim=-1)  
                    combined_features = F.relu(self.fc1(combined_features))
                    combined_features = F.relu(self.fc2(combined_features))
                    combined_features = self.fc3(combined_features)
                    combined_features = combined_features.view(32,-1,1).unsqueeze(-2).expand(-1,-1,self.n_agents,-1)
                    return combined_features
            else:
                for i in range(self.n_agents):

                    output = self.vector_net(
                        ego_trajectory_batch[:, i, :],
                        ego_reference_batch[:, i, :],
                        neighbour_trajectory_batch[:, i, :, :],
                        neighbour_reference_batch[:, i, :, :],
                        vectormap_batch[:, i, :]
                    )
                    outputs.append(output)
                if self.centralised:
                    combined_features = torch.cat(outputs, dim=-1) 
                    combined_features = F.relu(self.fc1(combined_features))
                    combined_features = F.relu(self.fc2(combined_features))
                    combined_features = self.fc3(combined_features)
                    combined_features = combined_features.unsqueeze(-2).expand(-1,self.n_agents,-1)
                    return combined_features


class CustomSequential(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules_list = nn.ModuleList(modules)
        
    def forward(self, *args):
        x = args
        for module in self.modules_list:
            if isinstance(x, tuple):
                x = module(*x)
            else:
                x = module(x)
        return x
    

