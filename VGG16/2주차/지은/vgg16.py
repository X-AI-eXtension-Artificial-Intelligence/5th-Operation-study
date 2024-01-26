import torch.nn as nn
import torch.nn.functional as F


def conv_2_block(in_dim,out_dim):
  #Sequential이란?
  #nn.Linear, nn.ReLU(활성화 함수) 같은 모듈들을 인수로 받아서
  #순서대로 정렬해놓고 입력값이 들어모면 순서대로 모듈을 실행해서 결과값을 리턴

  #이미지
  model = nn.Sequential(
      nn.Conv2d(in_dim, out_dim, kernel_size = 3, padding=1),
      nn.ReLU(),
      nn.Conv2d(out_dim, out_dim, kernel_size = 3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2,2)
  )
  return model

def conv_3_block(in_dim,out_dim):
  model = nn.Sequential(
      nn.Conv2d(in_dim,out_dim,kernel_size=3,padding=1),
      #이미지 크기에 대한 값은 들어가지 않음
      #only 채널 수
      nn.ReLU(),
      nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1),
      nn.ReLU(),
      nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2,2)
  )
  return model

#nn:신경망 계층(layer)과 거의 비슷한 Module 의 집합을 정의
#Module: 입력 텐서를 받고 출력 텐서를 계산
#nn.Module : 신경망 모델을 정의할 때 사용되는 기본 클래스 (레시피를 의미)

class VGG(nn.Module): #class VGG는 nn.Module을 상속함
  def __init__(self,base_dim,num_classes=10):
    super(VGG,self).__init__() #nn.Module의 생성자를 호출하여 초기화하는 역할
    #여기에서 VGG 클래스에 필요한 다양한 레이어들을 추가할 수 있음!

    #채널 수 관련
    self.feature = nn.Sequential(
        #base_dim = 64채널임
        conv_2_block(3,base_dim), #64채널
        conv_2_block(base_dim,2*base_dim), #64*2 = 128채널
        conv_3_block(2*base_dim,4*base_dim), #64*4 = 256채널
        conv_3_block(4*base_dim,8*base_dim), #64*8 = 512채널
        conv_3_block(8*base_dim, 8*base_dim) #64*8 = 512채널
    )

    #FC layer
    self.fc_layer = nn.Sequential(
        #flatten 시킨 것에 대한 값 계산
        nn.Linear(8*base_dim*2*2,4096), #512*1*1 <-> input : 224x224 -> 7*7*512
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096,1000),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(1000,num_classes) #다른 데이터셋이기 때문에, class 개수에 맞게 더 줄임
    )

  #이미지 x가 input으로 들어감
  def forward(self,x):
    x = self.feature(x)
    x = x.view(x.size(0),-1) #1차원으로 펼침(flatten)
    x = self.fc_layer(x)
    return x