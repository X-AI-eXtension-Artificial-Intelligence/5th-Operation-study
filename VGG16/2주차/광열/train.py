import torch
import torch.nn as nn
import torchvision.transforms as transforms

from vgg16 import VGG16
from torch.utils.data import DataLoader
from torchvision import datasets

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epoch = 10
learning_rate = 0.0002
batch_size = 32

transform = transforms.Compose(
    [transforms.ToTensor(), # [] 빼먹음
     transforms.Resize((224, 224)),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

train_data = datasets.ImageFolder('./Data/cat_dog/training_set',transform=transform) #  dataset는 공부 필요

train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)

model = VGG16(base_dim=64).to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate) # torch.optim, lr=


for i in range(num_epoch):
    for j,[image,label] in enumerate(train_loader):
        x = image.to(device)
        y = label.to(device)

        optimizer.zero_grad()

        output = model.forward(x)
        loss = loss_func(output,y)

        loss.backward()
        optimizer.step()
    
    if i%2==0:
        print(f'epcoh {i} loss : ',loss)

torch.save(model.state_dict(),"./train_model/VGG16_100.pth")

