import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernels=(3, 24), dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernels[0])
        self.act1 = nn.LeakyReLU(negative_slope=0.3)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernels[0])
        self.act2 = nn.LeakyReLU(negative_slope=0.3)
        
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernels[1], stride=2)
        self.act3 = nn.LeakyReLU(negative_slope=0.3)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        
        x = self.conv2(x)
        x = self.act2(x)
        
        x = self.conv3(x)
        x = self.act3(x)
        
        x = self.dropout(x)
        return x



class Attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels, 1, bias=True)
        self.act1 = nn.Tanh()
        
        self.conv2 = nn.Conv1d(in_channels, 1, 1, bias=False)
        self.act2 = nn.Softmax(dim=2)
        
    def forward(self, x):
        at = self.conv1(x)
        at = self.act1(at)
        
        at = self.conv2(at)
        at = self.act2(at)
        
        x = x * at
        return x.sum(dim=2)
    
    
class BiGRU(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, bidirectional=True):
        super().__init__()
        self.gru = nn.GRU(in_channels, out_channels, num_layers=num_layers, bidirectional=bidirectional)
        
    def forward(self, x):
        x = x.permute(2, 0, 1)
        x, _ = self.gru(x)
        x = x.permute(1, 2, 0)
        return x
    
    
class PhyChal2020Net(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        
        self.conv1 = ConvBlock(12, 12, kernels=(3, 24), dropout=dropout)
        self.conv2 = ConvBlock(12, 12, kernels=(3, 24), dropout=dropout)
        self.conv3 = ConvBlock(12, 12, kernels=(3, 24), dropout=dropout)
        self.conv4 = ConvBlock(12, 12, kernels=(3, 24), dropout=dropout)
        self.conv5 = ConvBlock(12, 12, kernels=(3, 48), dropout=dropout)
        self.gru = BiGRU(12, 12, 1, bidirectional=True)
        self.attention = Attention(24)
        self.bn = nn.BatchNorm1d(24)
        self.linear = nn.Linear(24, 9)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.gru(x)
        x = self.attention(x)
        x = self.bn(x)
        x = self.linear(x)
        return x