import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from vgg16 import VGG
import torch
import torch.nn as nn
from train import test_loader #New!!!!!


device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

#hyperparameter
batch_size = 32 
num_workers = 8 #GPU
learning_rate = 0.01
num_epoch = 100
ratio = 0.8 


# Transform 정의
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Train
model = VGG(base_dim=64).to(device)
model.load_state_dict(torch.load('./trained_model/VGG16_100.pth'))

# Test 결과
correct = 0
total = 0

model.eval() # 학습할 때만 필요했던 Dropout, Batchnorm등의 기능을 비활성화 해줌

with torch.no_grad(): #가중치 업데이트 x
  for image, label in test_loader:

    x = image.to(device)
    y = label.to(device)

    output = model.forward(x)
    #output의 크기는 (배치 크기) x (클래스의 개수)
    #ex) outputs = [[0.1, 0.4, 0.5], [0.2, 0.6, 0,2]] => 배치 크기 : 2, 클래스 : 3
    #torch.max는 최댓값과 최댓값의 위치를 반환하는데, 우리가 필요한 것은 최댓값 위치만 필요함
    #따라서, _로 처리해서 받지 않음 -> 최댓값의 위치(인덱스)만 total 에 저장
    #https://www.inflearn.com/questions/282058/%EB%AA%A8%EB%8D%B8-%ED%8F%89%EA%B0%80-%EB%B6%80%EB%B6%84-%EC%A7%88%EB%AC%B8%EB%93%9C%EB%A6%BD%EB%8B%88%EB%8B%A4
    _,output_index = torch.max(output,1)


    #전체 개수 -> 라벨의 개수
    #전체 개수를 알고 있음에도 이렇게 하는 이유 : batch_size, drop_last의 영향으로 몇몇 데이터가 잘린다?
    total += label.size(0) #개수 누적(총개수)
    #print(total) #테스트 이미지 개수


    correct += (output_index ==y).sum().float() #누적(맞으면 1, 틀리면 0으로 합산)
    #print(correct)

  print("Accuracy of Test Data: {}%".format(100*correct/total))