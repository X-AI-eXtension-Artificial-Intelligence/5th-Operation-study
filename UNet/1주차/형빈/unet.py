import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from torchvision import transforms,datasets

## build architecture
class UNet(nn.Module):
    def __init__(self):
        super(UNet,self).__init__()

        # Conv + batch norm + relu
        def ConvBlock(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=True):
            layers=[]
            layers += [nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                                 kernel_size=kernel_size,stride=stride,padding=padding,
                                 bias=bias)]
            layers+=[nn.BatchNorm2d(num_features=out_channels)]
            layers+=[nn.ReLU()]

            cbrBlock = nn.Sequential(*layers)

            return cbrBlock
        
        # contracting path

        self.contract1_1 = ConvBlock(in_channels=1,out_channels=64)
        self.contract1_2 = ConvBlock(in_channels=64,out_channels=64)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.contract2_1 = ConvBlock(in_channels=64,out_channels=128)
        self.contract2_2 = ConvBlock(in_channels=128,out_channels=128)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.contract3_1 = ConvBlock(in_channels=128,out_channels=256)
        self.contract3_2 = ConvBlock(in_channels=256,out_channels=256)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.contract4_1 = ConvBlock(in_channels=256,out_channels=512)
        self.contract4_2 = ConvBlock(in_channels=512,out_channels=512)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.contract5_1 = ConvBlock(in_channels=512,out_channels=1024)

        # expansive path

        self.expand5_1 = ConvBlock(in_channels=1024,out_channels=512)
        
            ## upconv 2x2
        self.upConv4 = nn.ConvTranspose2d(in_channels=512,out_channels=512,
                                          kernel_size=2,stride=2,padding=0,bias=True)
        
        self.expand4_2 = ConvBlock(in_channels=512*2,out_channels=512)
        self.expand4_1 = ConvBlock(in_channels=512,out_channels=256)

        self.upConv3 = nn.ConvTranspose2d(in_channels=256,out_channels=256,
                                          kernel_size=2,stride=2,padding=0,bias=True)
        
        self.expand3_2 = ConvBlock(in_channels=256*2,out_channels=256)
        self.expand3_1 = ConvBlock(in_channels=256,out_channels=128)

        self.upConv2 = nn.ConvTranspose2d(in_channels=128,out_channels=128,
                                          kernel_size=2,stride=2,padding=0,bias=True)
        
        self.expand2_2 = ConvBlock(in_channels=2 * 128, out_channels=128)
        self.expand2_1 = ConvBlock(in_channels=128, out_channels=64)

        self.upConv1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.expand1_2 = ConvBlock(in_channels=2 * 64, out_channels=64)
        self.expand1_1 = ConvBlock(in_channels=64, out_channels=64)

        self.fc = nn.Conv2d(in_channels=64,out_channels=1,kernel_size=1,stride=1,padding=0,bias=True)

    # forward func
    def forward(self,x):
        contract1_1 = self.contract1_1(x)
        contract1_2 = self.contract1_2(contract1_1)
        pool1 = self.pool1(contract1_2)

        contract2_1 = self.contract2_1(pool1)
        contract2_2 = self.contract2_2(contract2_1)
        pool2 = self.pool2(contract2_2)

        contract3_1 = self.contract3_1(pool2)
        contract3_2 = self.contract3_2(contract3_1)
        pool3 = self.pool3(contract3_2)

        contract4_1 = self.contract4_1(pool3)
        contract4_2 = self.contract4_2(contract4_1)
        pool4 = self.pool4(contract4_2)

        contract5_1 = self.contract5_1(pool4)

        expand5_1 = self.expand5_1(contract5_1)

        upConv4 = self.upConv4(expand5_1)
        cat4 = torch.cat((upConv4,contract4_2),dim=1)
        expand4_2 = self.expand4_2(cat4)
        expand4_1 = self.expand4_1(expand4_2)

        upConv3 = self.upConv3(expand4_1)
        cat3 = torch.cat((upConv3,contract3_2),dim=1)
        expand3_2 = self.expand3_2(cat3)
        expand3_1 = self.expand3_1(expand3_2)

        upConv2 = self.upConv2(expand3_1)
        cat2 = torch.cat((upConv2, contract2_2),dim=1)
        expand2_2 = self.expand2_2(cat2)
        expand2_1 = self.expand2_1(expand2_2)

        upConv1 = self.upConv1(expand2_1)
        cat1 = torch.cat((upConv1,contract1_2),dim=1)
        expand1_2 = self.expand1_2(cat1)
        expand1_1 = self.expand1_1(expand1_2)

        x = self.fc(expand1_1)

        return x



    