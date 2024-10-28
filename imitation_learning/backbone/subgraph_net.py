import torch
import torch.nn as nn
import torch.nn.functional as F  # useful stateless functions
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# class SubgraphNet_Layer(nn.Module):
#     def __init__(self, input_channels = 128, hidden_channels = 64):
#         super().__init__()
#         self.fc = nn.Linear(input_channels, hidden_channels)
#         nn.init.kaiming_normal(self.fc.weight)

#     def forward(self, input):
#         hidden = self.fc(input).unsqueeze(0)                           # 一个全连接层,unsqueeze增加一维 torch.Size([r, c]) -> torch.Size([1, r, c])
#         encode_data = F.relu(F.layer_norm(hidden, hidden.size()[1:]))  # layer norm and relu
#         kernel_size = encode_data.size()[1]                            # 30
#         maxpool = nn.MaxPool1d(kernel_size)                            # 最大值池化
#         polyline_feature = maxpool(encode_data.transpose(1,2)).squeeze()
#         polyline_feature = polyline_feature.repeat(kernel_size, 1)
#         output = torch.cat([encode_data.squeeze(), polyline_feature], 1) # 拼接relu结果和池化结果，输出shape(r,2*c)
#         return output

# class SubgraphNet(nn.Module):
#     def __init__(self, input_channels):
#         super().__init__()
#         self.sublayer1 = SubgraphNet_Layer(input_channels)
#         self.sublayer2 = SubgraphNet_Layer()
#         self.sublayer3 = SubgraphNet_Layer() #output = 128

#     def forward(self, input):           # input.shape -> torch.Size([30, 6]) torch.Size([18, 8]),
#         out1 = self.sublayer1(input)    # 调用SubgraphNet_Layer.forward(input)，  out -> (30, 128)
#         out2 = self.sublayer2(out1)     # out2 -> (128, 128)
#         out3 = self.sublayer3(out2)     # out3 -> (128, 128)
#         kernel_size = out3.size()[0]    # 128
#         maxpool = nn.MaxPool1d(kernel_size)
#         polyline_feature = maxpool(out3.unsqueeze(1).transpose(0,2)).squeeze()  # polyline_feature.shape -> torch.Size([128])
#         return polyline_feature   
    

# class SubgraphNet_Layer(nn.Module):
#     def __init__(self, input_channels=256, hidden_channels=128):
#         super().__init__()
#         self.fc = nn.Linear(input_channels, 64)
#         nn.init.kaiming_normal_(self.fc.weight)
#         self.fc2 = nn.Linear(64, 128)
#         nn.init.kaiming_normal_(self.fc2.weight)

#     def forward(self, input):
#         x = self.fc(input)  # 直接对整个批次应用全连接层
#         x = F.relu(F.layer_norm(x, x.size()[1:]))  # 应用LayerNorm和ReLU
#         x = self.fc2(x)
#         encode_data = F.relu(F.layer_norm(x, x.size()[1:])) 

#         # 使用自适应最大池化来处理不同长度的序列
#         maxpool = nn.AdaptiveMaxPool1d(1)  # 输出大小为1
#         pooled = maxpool(encode_data.transpose(1, 2)).squeeze(2)  # 在最后一个维度上池化并移除多余的维度

#         # 重复扩展以匹配输入维度，然后和原始编码数据拼接
#         polyline_feature = pooled.unsqueeze(1).repeat(1, encode_data.size(1), 1)
#         output = torch.cat([encode_data, polyline_feature], dim=2)
#         return output

# class SubgraphNet(nn.Module):
#     def __init__(self, input_channels):
#         super().__init__()
#         self.sublayer1 = SubgraphNet_Layer(input_channels)
#         self.sublayer2 = SubgraphNet_Layer()
#         self.sublayer3 = SubgraphNet_Layer()

#     def forward(self, input):
#         out1 = self.sublayer1(input)
#         out2 = self.sublayer2(out1)
#         out3 = self.sublayer3(out2)
#         maxpool = nn.AdaptiveMaxPool1d(1)
#         # 需要首先交换维度，因为AdaptiveMaxPool1d预期序列维度是最后一个维度
#         pooled = maxpool(out3.transpose(1, 2))  # [batch_size, 128, 1]
#         polyline_feature = pooled.squeeze(2)  # 去除最后一个维度，形状为 [batch_size, 128]

#         return polyline_feature
    
# class SubgraphNet4state(nn.Module):
#     def __init__(self, input_channels = 4):
#         super().__init__()
#         self.fc = nn.Linear(input_channels, 64)
#         nn.init.kaiming_normal_(self.fc.weight)
#         self.fc2 = nn.Linear(64, 256)
#         nn.init.kaiming_normal_(self.fc2.weight)
#     def forward(self, input):
#         x = self.fc(input)
#         x = F.relu(F.layer_norm(x, x.size()[1:]))
#         x = F.relu(self.fc2(x)).squeeze(1)
#         return x
    

# class SubgraphNet_Layer(nn.Module):
#     def __init__(self, input_channels=256, hidden_channels=128):
#         super().__init__()
#         self.fc = nn.Linear(input_channels, 64)
#         nn.init.kaiming_normal_(self.fc.weight)
#         self.fc2 = nn.Linear(64, 128)
#         nn.init.kaiming_normal_(self.fc2.weight)

#     def forward(self, input):
#         x = self.fc(input)  # 直接对整个批次应用全连接层
#         x = F.relu(F.layer_norm(x, x.size()[1:]))  # 应用LayerNorm和ReLU
#         x = self.fc2(x)
#         encode_data = F.relu(F.layer_norm(x, x.size()[1:])) 

#         # 使用自适应最大池化来处理不同长度的序列
#         maxpool = nn.AdaptiveMaxPool1d(1)  # 输出大小为1
#         pooled = maxpool(encode_data.transpose(1, 2)).squeeze(2)  # 在最后一个维度上池化并移除多余的维度

#         # 重复扩展以匹配输入维度，然后和原始编码数据拼接
#         polyline_feature = pooled.unsqueeze(1).repeat(1, encode_data.size(1), 1)
#         output = torch.cat([encode_data, polyline_feature], dim=2)
#         return output

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

        # concat with previous feature
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

        pooled = maxpool(out3.transpose(1, 2)) 
        polyline_feature = pooled.squeeze(2)  

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