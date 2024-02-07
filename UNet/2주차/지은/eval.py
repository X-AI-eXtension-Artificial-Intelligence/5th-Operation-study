
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets 

from unet import UNet

#hypterparameter 설정
lr = 1e-3
batch_size=4
num_epoch=100

data_dir ='./datasets'
ckpt_dir ='./checkpoint'
log_dir='./log'
result_dir = './results'

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#데이터 로더 구현
class Dataset(torch.utils.data.Dataset):
    def __init__(self,data_dir,transform=None): #argument 값 선언 
        self.data_dir = data_dir
        self.transform = transform
        
        lst_data = os.listdir(self.data_dir)
        
        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]
        
        lst_label.sort()
        lst_input.sort()
        
        self.lst_label = lst_label
        self.lst_input = lst_input
        
    def __len__(self): #data length 확인
        return len(self.lst_label)
    
    def __getitem__(self, index): #데이터 불러오기 
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))
        
        label = label/255.0
        input = input/255.0
        
        #Neural Network에 들어가야 하는 input은 3개의 access를 가져야 함
        #x,y,channel
        
        #이미지와 레이블의 차원 : 2차원일 경우,
        #새로운 채널(축) 생성
        #(3,4) -> [:,:,np.newaxis] -> (3,4,1) 세로, 가로 , 넓이 ????????????????
        if label.ndim ==2:
            label=label[:,:,np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]
        
        data = {'input': input, 'label': label}

        if self.transform:
            data = self.transform(data)

        return data


#Transform 직접 정의

#Tensor로 변경
class ToTensor(object):
    def __call__(self,data):
        label, input = data['label'], data['input']
        
        #Image의 numpy 차원 : Y,X,CH
        #Image의 tensor 차원 : CH,Y,X
        label = label.transpose((2,0,1)).astype(np.float32) 
        input = input.transpose((2,0,1)).astype(np.float32)
        
        #numpy를 tensor로 넘겨주는 함수 : from_numpy
        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data

#train : -1(어두운 부분) ~ 1(밝은 부분) 사이의 범위를 갖는다.
#label : 0(어두운 부분) ~ 1(밝은 부분)
class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        #라벨은 건들 ㄴㄴ
        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}

        return data

class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5:
            label = np.fliplr(label) #오왼 반전 
            input = np.fliplr(input)
        
        #50% 확률로
        if np.random.rand() > 0.5:
            label = np.flipud(label) #위아래 반전
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data

# 훈련을 위한 Transform과 DataLoader 불러오기
transform = transforms.Compose([Normalization(mean=0.5, std=0.5),  ToTensor()])

# num_workers : GPU 대수 * 4
dataset_test = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)


# 네트워크 생성하기
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = UNet().to(device)

#손실 함수 정의
#Sigmoid layer + BCELoss(Binaray Classification)의 조합
# 1 or 0이 나오도록
#https://cvml.tistory.com/26
fn_loss = nn.BCEWithLogitsLoss().to(device)

#Optimizer 설정하기
optim = torch.optim.Adam(net.parameters(),lr=lr)

#추가 설정
num_data_test= len(dataset_test)

num_batch_test= np.ceil(num_data_test/batch_size)

# 그 밖에 부수적인 functions 설정하기
#tensor에서 numpy로
#batch, channel, y, x -> batch, y,x,channel 
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
#denormalize
fn_denorm = lambda x, mean, std: (x * std) + mean
#binary class 기준 설정
fn_class = lambda x: 1.0 * (x > 0.5)

#네트워크 저장
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        
    torch.save({'net': net.state_dict(), 'optim': optim.state_dict(),},
               './%s/model_epoch%d.pth'% (ckpt_dir, epoch))

## 네트워크 불러오기
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch
        
    

#네트워크 학습
net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

with torch.no_grad():
      net.eval()
      loss_arr = []

      for batch, data in enumerate(loader_test, 1):
          # forward pass
          label = data['label'].to(device)
          input = data['input'].to(device)

          output = net(input)

          # 손실함수 계산하기
          loss = fn_loss(output, label)

          loss_arr += [loss.item()]

          print("TEST: BATCH %04d / %04d | LOSS %.4f" %
                (batch, num_batch_test, np.mean(loss_arr)))

          
print("LOSS %.4f" %
        (np.mean(loss_arr)))


#https://www.youtube.com/watch?v=igvk1W1JtHA&t=349s
#한요섭 님께 무한한 감사를. 