import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import random
import copy
from sklearn.model_selection import train_test_split
import os

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__() 
        self.out_channels = out_channels
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels))
        
        # initalize all the weights of every layer properly 
        self.model.apply(self._initWeights)
        
    def forward(self, x):
        return self.model(x) 
    
    def _initWeights(self, m):
        # init weights of convolution layers according to kaiming initialization
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode = 'fan_out',nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight,1)
            nn.init.constant_(m.bias, 0)

class ResidualBlocks(nn.Module):
    def __init__(self, num_layers_perBlock):
        super(ResidualBlocks, self).__init__() 
        
        # create a module list to hold all the blocks 
        self.container = nn.ModuleList() 
        # need to store a couple projection shortcuts -> each block
        # excluding the first has one connection shortcut to connect the shortcut connection from prev
        # block to current block
        self.weighted_skipConnects = nn.ModuleDict()
        # 3 blocks of residual connections, each with a different amount of filters
        NumFilters = {0:16, 1:32, 2:64}
        weighted_skip_connectCounter = 0
        for i in range(3):
            # for every layer in the current residual block (we have 2n total layers in every block)
            for j in range(2*num_layers_perBlock):
                in_channels = 16 if not self.container else self.container[-1].out_channels
    
                # we need a stride of two for the first convolution operation of the second batch of convolution operations
                # and for the third batch of convolution operations to downsample the input by 2 in terms of H and W
                stride = 2 if (i >0 and j==0) else 1
                # padding is kept at a constant value of 1 for every single conv operation, even when we're downsampling
                # keeps operations symmetric - we upsample the number of filters by a factor of 2 and downsample the H and W by
                # a factor of 2
                
                self.container.append(ConvBlock(in_channels, out_channels=NumFilters[i], kernel_size =3, stride = stride, padding = 1))
                
                if i >0 and j == 0:
                    self.weighted_skipConnects['WeightedSkipConnect'+str(weighted_skip_connectCounter)] = nn.Sequential(nn.Conv2d(in_channels = in_channels, out_channels = NumFilters[i], kernel_size=1, stride=2, padding=0),
                                                                     nn.BatchNorm2d(NumFilters[i]))
                    weighted_skip_connectCounter += 1 
                
    def forward(self, x):
        # This is where the actual skip connections happen
        # the architecture is technically the same as a plain deep net
        # but during the forward pass (and the backward pass too), we take 
        # the input to the even layers and fast forward it a count of 2 layers
        # down from the current layer 
        skip_connection = x.clone() 
        curr_x = x 
        weighted_skip_connectCounter = 0
        for i in range(len(self.container)): 
            # pass the current x through a conv operation and
            # batch norm operation 
            curr_x = self.container[i](curr_x)
            # every two steps, we add a skip connection to the curr_x and then pass it
            # through the ReLU 
            if i%2 == 1:
                # use a 1x1 convolution with a stride of 2 to downsample in terms of height and
                # width, and upsample in terms of number of channels ONLY if the shapes dont match 
                if skip_connection.shape != curr_x.shape:
                    skip_connection = self.weighted_skipConnects['WeightedSkipConnect'+str(weighted_skip_connectCounter)](skip_connection)
                    weighted_skip_connectCounter += 1 
                curr_x = curr_x.add(skip_connection)
                # curr_x becomes new skip connection 
                skip_connection = curr_x.clone() 
                
            # didnt include ReLU in the conv blocks because we add the
            # skip connection after the batch norm operation of the conv block, and then 
            # activate 
            curr_x = nn.ReLU()(curr_x)
        return curr_x

    
class ResNet(nn.Module):
    def __init__(self, depth = 20):
        torch.manual_seed(21)
        super(ResNet,self).__init__()
        n = (depth-2)//6
        assert type(n) is int, "Num blocks per stack not equal to integer"
        # This will return a nn.Module object with all of the residual skip blocks 
        # used in the network (total being 6*n of these blocks)
        residualBlockConnections_6n = ResidualBlocks(n)
        # Then we just have a Conv2D operation, followed by the entire residual section block
        # followed by a global avg pool, then a 10 way softmax 
        self.model = nn.Sequential(
            ConvBlock(3, 16, kernel_size = 3, stride=1, padding=1),
            nn.ReLU(),
            residualBlockConnections_6n,
            nn.AvgPool2d(kernel_size = 8),
            nn.Flatten(),
            # cross entropy loss expects raw logits of shape (N,C)
            nn.Linear(in_features = 64, out_features = 10))
        
    def forward(self, x):
        return self.model(x)
            