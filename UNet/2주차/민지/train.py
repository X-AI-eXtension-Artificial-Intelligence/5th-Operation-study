import numpy as numpy
## train
import torch
import torchvision
import torch.nn as nn
# import deeplake
from torchvision import transforms
from torch.utils.data import DataLoader
from Unet import UNet

import deeplake
train_dset = deeplake.load("hub://activeloop/drive-train")
train_loader = train_dset.pytorch(num_workers=0, batch_size=4, shuffle=True)

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5], std=[0.5])]) 
train_dset = torchvision.datasets.ImageFolder(root='C:/Users/SAMSUNG/Desktop/X-AI/코드구현/data/cats/train/resized(32)', transform=transform)

train_loader = DataLoader(dataset=train_dset, batch_size= 32, shuffle=True, drop_last= False)
num_class = len(train_dset.classes)



model = UNet()
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-06)


loss_arr = []
epoch = 5
model.train()

for i in range(epoch):
    for j, [image, label] in enumerate(train_loader):
        x = image
        y = label
        
        optimizer.zero_grad()
        output = model.forward(x)

        loss = loss_func(output, y)
        loss.backward()
        optimizer.step()
        
        if j % 10 ==0:
            print("loss :", loss)
            loss_arr.append(loss.cpu().detach().numpy())



torch.save(model, 'UNet_model.pth')