from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torchvision


# data loader
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5], std=[0.5]), 
                                transforms.Resize((32,32))]) 
test_dset = torchvision.datasets.ImageFolder(root='data/cats/test/resized(32)', transform=transform)
test_loader = DataLoader(dataset=test_dset, batch_size=32, shuffle=True, drop_last=False)



correct = 0
total = 0

# 학습한 모델 불러오기
model = torch.load('VGG_model.pth')


# inference
model.eval()

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