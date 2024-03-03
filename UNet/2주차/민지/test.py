import numpy as numpy
from torch.utils.data import DataLoader
import torch
import torchvision
# import deeplake
from torchvision import transforms
from Unet import UNet

import deeplake


test_dset = deeplake.load("hub://activeloop/drive-test")
test_loader = test_dset.pytorch(num_workers=0, batch_size=4, shuffle=True)

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5], std=[0.5])]) 
test_dset = torchvision.datasets.ImageFolder(root='C:/Users/SAMSUNG/Desktop/X-AI/코드구현/data/cats/test/resized(32)', transform=transform)
test_loader = DataLoader(dataset=test_dset, batch_size=32, shuffle=True, drop_last=False)

num_class = len(test_dset.classes)

model = UNet()
model.load_state_dict(torch.load('UNet_model.pth'))
model.eval()
total = 0
correct = 0
# 인퍼런스 모드 :  no_grad 
with torch.no_grad():
    # 테스트로더에서 이미지와 라벨 불러와서
    for image,label in test_loader:
        x = image
        y= label

        # 모델에 데이터 넣고 결과값 얻기
        output = model.forward(x)
        _,output_index = torch.max(output,1)

        
        # 전체 개수 += 라벨의 개수
        total += label.size(0)
        correct += (output_index == y).sum().float()
    
    # 정확도 도출
    print("Accuracy of Test Data: {}%".format(100*correct/total))