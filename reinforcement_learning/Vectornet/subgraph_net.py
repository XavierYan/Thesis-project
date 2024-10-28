import torch
import torch.nn as nn
import torch.nn.functional as F  # useful stateless functions
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
    

class SubgraphNet_Layer(nn.Module):
    def __init__(self, input_channels=256, hidden_channels=128):
        super().__init__()
        self.fc = nn.Linear(input_channels, 64)
        nn.init.kaiming_normal_(self.fc.weight)
        self.fc2 = nn.Linear(64, 128)
        nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, input):
        x = self.fc(input)  
        x = F.relu(F.layer_norm(x, x.size()[1:])) 
        x = self.fc2(x)
        encode_data = F.relu(F.layer_norm(x, x.size()[1:])) 


        maxpool = nn.AdaptiveMaxPool1d(1) 
        pooled = maxpool(encode_data.transpose(1, 2)).squeeze(2)  

        polyline_feature = pooled.unsqueeze(1).repeat(1, encode_data.size(1), 1)
        output = torch.cat([encode_data, polyline_feature], dim=2)
        return output

class SubgraphNet(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.sublayer1 = SubgraphNet_Layer(input_channels)
        self.sublayer2 = SubgraphNet_Layer()
        self.sublayer3 = SubgraphNet_Layer()

    def forward(self, input):
        out1 = self.sublayer1(input)
        out2 = self.sublayer2(out1)
        out3 = self.sublayer3(out2)
        maxpool = nn.AdaptiveMaxPool1d(1)
        pooled = maxpool(out3.transpose(1, 2))  # [batch_size, 128, 1]
        polyline_feature = pooled.squeeze(2)  # [batch_size, 128]

        return polyline_feature
    
class SubgraphNet4state(nn.Module):
    def __init__(self, input_channels = 4):
        super().__init__()
        self.fc = nn.Linear(input_channels, 64)
        nn.init.kaiming_normal_(self.fc.weight)
        self.fc2 = nn.Linear(64, 256)
        nn.init.kaiming_normal_(self.fc2.weight)
    def forward(self, input):
        x = self.fc(input)
        x = F.relu(F.layer_norm(x, x.size()[1:]))
        x = F.relu(self.fc2(x)).squeeze(1)
        return x