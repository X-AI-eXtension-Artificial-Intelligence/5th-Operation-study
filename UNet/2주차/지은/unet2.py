import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import numpy as np

from torchvision import transforms

#https://sd118687.tistory.com/8 (Mirroring extrapolate)

class UNet(nn.Module):
    #copy and crop 함수 정의
    #batch_size, channels, height, width
    def copy_and_crop(self, input, output):
        output_size = output.size()[2:]
        input_size = input.size()[2:]
        
        #수축 부분과 확장 부분의 w,h 크기 비교해서 가장 작은 값의 사이즈 추출
        crop_size = [min(i,o) for i,o in zip(input_size, output_size)]
        
        #자를 시작점 선정 (그림참고)
        crop_start = [int((i - c) / 2) for i, c in zip(input_size, crop_size)]
        
        #인덱싱은 차원을 낮추고, 슬라이싱은 차원을 보존!!!!!!!!!!!!!
        cropped_input = input[:, :, crop_start[0]:crop_start[0] + crop_size[0], crop_start[1]:crop_start[1] + crop_size[1]]
        
        return cropped_input
        
    def __init__(self):
        super(UNet, self).__init__() #.__init__()
        
        def act(in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True):
            layers=[]
            layers+=[nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride, padding=padding,
                               bias=bias)]
            layers+=[nn.BatchNorm2d(num_features=out_channels)]#nn. , []
            layers+=[nn.ReLU()]
            
            #nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
            #                   kernel_size=kernel_size, stride=stride, padding=padding,
            #                   bias=bias), 
            #                   nn.BatchNorm2d(num_features=out_channels),
            #                   nn.ReLU())              
            results = nn.Sequential(*layers) #* (가변인자)
            
            #print(layers)
            #[Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1)), BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU()]
            
            return results
        
        #Contractive path
        self.con1_1 = act(in_channels=1, out_channels=64)
        self.con1_2 = act(in_channels=64, out_channels=64)
        
        self.con_pool1 = nn.MaxPool2d(kernel_size=2) #2,2
        
        self.con2_1 = act(in_channels=64, out_channels=128)
        self.con2_2 = act(in_channels=128, out_channels=128)
        
        self.con_pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.con3_1 = act(in_channels=128,out_channels=256)
        self.con3_2 = act(in_channels=256, out_channels=256)
        
        self.con_pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.con4_1 = act(in_channels=256, out_channels=512)
        self.con4_2 = act(in_channels=512, out_channels=512)
        
        self.con_pool4 = nn.MaxPool2d(kernel_size=2)
        
        self.con5_1 = act(in_channels=512, out_channels=1024)
        
        
        #expansive path
        self.ex5_1 = act(in_channels=1024, out_channels=512) #그림이 참..
        
        self.ex_unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                             kernel_size=2, stride=2, padding=0, bias=True)
        
        self.ex4_2 = act(in_channels=512*2,out_channels=512)
        self.ex4_1 = act(in_channels=512, out_channels=256)
        
        self.ex_unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                              kernel_size=2, stride=2, padding=0, bias=True)
        
        self.ex3_2 = act(in_channels=256*2, out_channels=256)
        self.ex3_1 = act(in_channels=256, out_channels=128)
        
        self.ex_unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                              kernel_size=2, stride=2, padding=0, bias=True)
        
        self.ex2_2 = act(in_channels=128*2, out_channels=128)
        self.ex2_1 = act(in_channels=128, out_channels=64)
        
        self.ex_unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                              kernel_size=2, stride=2, padding=0, bias=True)
        
        self.ex1_2 = act(in_channels=64*2, out_channels=64) #반대로씀 ;
        self.ex1_1 = act(in_channels=64, out_channels=64)
        
        
        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0,
                            bias = True) #bais 오타
        
        
    def forward(self,input):
        con1_1 = self.con1_1(input)
        con1_2 = self.con1_2(con1_1)
        con_pool1 = self.con_pool1(con1_2)
        
        con2_1 = self.con2_1(con_pool1)
        con2_2 = self.con2_2(con2_1)
        con_pool2 = self.con_pool2(con2_2)
        
        con3_1 = self.con3_1(con_pool2)
        con3_2 = self.con3_2(con3_1)
        con_pool3 = self.con_pool3(con3_2)
        
        con4_1 = self.con4_1(con_pool3)
        con4_2 = self.con4_2(con4_1)
        con_pool4 = self.con_pool4(con4_2)
        
        con5_1 = self.con5_1(con_pool4)
        
        ex5_1 = self.ex5_1(con5_1)
        
        #print(con1_1.size(), con1_2.size(), con_pool1.size(), con2_1.size(), con2_2.size(), con_pool2.size(), con3_1.size(), con3_2.size(), con_pool3.size(), con4_1.size(), con4_2.size(), con_pool4.size(), con5_1.size(), ex5_1.size())

        ex_unpool4 = self.ex_unpool4(ex5_1)
        cropped_con4_2 = self.copy_and_crop(con4_2,ex_unpool4)
        cat4= torch.cat((ex_unpool4,cropped_con4_2),dim=1)
        ex4_2 = self.ex4_2(cat4)
        ex4_1 = self.ex4_1(ex4_2)
        
        #print(ex_unpool4.size(),cropped_con4_2.size(), cat4.size(), ex4_2.size(),ex4_1.size() )
        
        ex_unpool3 = self.ex_unpool3(ex4_1)
        cropped_con3_2 = self.copy_and_crop(con3_2, ex_unpool3)
        cat3 = torch.cat((ex_unpool3, cropped_con3_2), dim=1)
        ex3_2 = self.ex3_2(cat3)
        ex3_1 = self.ex3_1(ex3_2)
        
        #print(ex_unpool3.size(),cropped_con3_2.size(), cat3.size(), ex3_2.size(),ex3_1.size() )
        
        ex_unpool2 = self.ex_unpool2(ex3_1)
        cropped_con2_2 = self.copy_and_crop(con2_2, ex_unpool2)
        cat2 = torch.cat((ex_unpool2, cropped_con2_2),dim=1)
        ex2_2 = self.ex2_2(cat2)
        ex2_1 = self.ex2_1(ex2_2)
        
        #print(ex_unpool2.size(),cropped_con2_2.size(), cat2.size(), ex2_2.size(),ex2_1.size() )
        
        ex_unpool1 = self.ex_unpool1(ex2_1)
        cropped_con1_2 = self.copy_and_crop(con1_2, ex_unpool1)
        cat1 = torch.cat((ex_unpool1, cropped_con1_2),dim=1)
        ex1_2 = self.ex1_2(cat1)
        ex1_1 = self.ex1_1(ex1_2)
                
        x = self.fc(ex1_1)
        
        #print(ex_unpool1.size(),cropped_con1_2.size(), cat1.size(),ex1_2.size(),ex1_1.size(),x.size())
        
        return x
        
        # Must be a tensor with equal size along the class dimension to the number of classes.
        #BCELoss는 output과 target 사이즈가 같아야 한다고 함..
        #귀찮으니 패스~
        #https://m.blog.naver.com/kee_pee/221575076301
            