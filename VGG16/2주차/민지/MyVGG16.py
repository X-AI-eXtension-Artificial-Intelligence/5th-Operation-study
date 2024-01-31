import numpy as np
import os
from random import randint
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from glob import glob

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def conv_2_block(in_channel, out_channel):
    model = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1), 
        nn.ReLU(),
        nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1), 
        nn.ReLU(),
        nn.MaxPool2d(2,2) # kernel_size, stride_size => 2x2 MaxPooling layer (stride=2)
    )
    return model


def conv_3_block(in_channel, out_channel):
    model = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
        )
    return model



class MyVGG(nn.Module):
    def __init__(self,base_channel_dim=64, num_class=10):
        super(MyVGG, self).__init__()
        
        self.features = nn.Sequential(
            conv_2_block(3, base_channel_dim),
            conv_2_block(base_channel_dim, base_channel_dim*2),
            conv_3_block(base_channel_dim*2, base_channel_dim*4),
            conv_3_block(base_channel_dim*4, base_channel_dim*8),
            conv_3_block(base_channel_dim*8, base_channel_dim*8),
                                 )
        
        self.classifier = nn.Sequential(
            nn.Linear(base_channel_dim*1*1*8, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_class)
            ### 변경. 내 데이터 class 수는 10이므로!

            )
        ## softmax를 작성하지 않는 이유
        # Pytorch의 nn.CrossEntropyLoss는 log softmax와 negative log likelihood를 결합하여 구현되어 있음
        # softmax를 사용하고 싶으면 softmax를 거친 결과에 log를 취한 후, nn.NLLLoss에 입력해야 함 (그렇게 안해도 결과는 같긴 함)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) # 배치는 keep -> 열을 h*w로  [32,3,3] -> [32,9]
        x = self.classifier(x)
        return x

