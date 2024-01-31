import torch
import torchvision
import torch.nn as nn
from PIL import Image
from MyVGG16 import MyVGG

from torchvision import transforms
from torch.utils.data import DataLoader


# ## 데이터 크기 변경  :  224x224 -> 32x32
# def resize_images(input_path, output_path, target_size=(32,32)):
#     for file in glob(os.path.join(input_path, '*/*')): # glob - 해당 경로에 있는 모든 파일에 대한 작업 처리. '/'- 모든 파일과 서브디렉토리를 나타내는 패턴
#         file = file.replace('\\','/')
#         img = Image.open(file).resize(target_size)
  
#         o_path = output_path + "resized(32)/"+file.split('/')[-2]
#         os.makedirs(o_path, exist_ok=True)

#         img.save(o_path+f"/{file.split('/')[-1]}")



transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5], std=[0.5]), 
                                transforms.Resize((32,32))]) 
# 각 픽셀의 RGB 픽셀 범위는 0~255임. 이를 각각 Normalize해주는 것
# transforms.Normalize(mean=[0.5], std=[0.5])]) ==> 색상이 표준화됨

train_dset = torchvision.datasets.ImageFolder(root='data/cats/train/resized(32)', transform=transform)
train_loader = DataLoader(dataset=train_dset, batch_size= 32, shuffle=True, drop_last= False)

num_class = len(train_dset.classes)
# img size = 224x224


model = MyVGG()
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
        
    #if i % 10 ==0:
    print("loss :", loss)
    loss_arr.append(loss.cpu().detach().numpy())



torch.save(model, 'VGG_model.pth')