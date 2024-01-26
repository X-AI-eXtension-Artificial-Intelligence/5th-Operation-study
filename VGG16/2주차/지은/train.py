import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from vgg16 import VGG
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split

#hyperparameter
batch_size = 32 
num_workers = 8 #GPU
learning_rate = 0.01
num_epoch = 100
ratio = 0.8 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Transform 정의
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Data
path2data = './data'
# ImageFolder를 통해 불러옴
dataset = ImageFolder(root=path2data,            
                      transform=transform)
# Train, Test 나누기
train_size = int(ratio * len(dataset))
test_size = len(dataset) - train_size
#print(f'total: {len(dataset)}\ntrain_size: {train_size}\ntest_size: {test_size}')

# random_split으로 8:2의 비율로 train / test 세트를 분할
train_data, test_data = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_data, 
                          batch_size=batch_size,
                          shuffle=True, 
                          num_workers=num_workers
                         )
test_loader = DataLoader(test_data, 
                         batch_size=batch_size,
                         shuffle=False, 
                         num_workers=num_workers
                        )
classes = ('AbdomenCT', 'BreastMRI', 'CXR', 'ChestCT', 'Hand', 'HeadCT')



# Train
model = VGG(base_dim=64).to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)


loss_arr = []

def train():
    for i in range(num_epoch):
        for j, [image,label] in enumerate(train_loader):
            x = image.to(device)
            y_ = label.to(device)

            optimizer.zero_grad() #초기화
            output = model.forward(x) #순전파
            loss = loss_func(output,y_) #loss_func
            loss.backward() #역전파
            optimizer.step() #가중치 업데이트

        if i%10 ==0: #loss 출력
            print(loss)
            #detach => requies_grad = False로 설정, gradient의 전파를 멈춤
            #https://iambeginnerdeveloper.tistory.com/211
            loss_arr.append(loss.cpu().detach().numpy()) #loss_arr에 추가
if __name__ == '__main__':
    train() #https://kangjik94.tistory.com/38

    

torch.save(model.state_dict(), "./trained_model/VGG16_100.pth")