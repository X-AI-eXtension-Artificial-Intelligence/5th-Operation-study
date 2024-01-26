import torch
import torchvision.transforms as transforms

from vgg16 import VGG16
from torch.utils.data import DataLoader
from torchvision import datasets

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 32

transform = transforms.Compose(
    [transforms.ToTensor(), # [] 빼먹음
     transforms.Resize((224, 224)),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

test_data = datasets.ImageFolder('./Data/cat_dog/test_set',transform=transform) #  dataset는 공부 필요

test_loader = DataLoader(test_data,batch_size=batch_size)

model = VGG16(base_dim=64).to(device)
model.load_state_dict(torch.load("./train_model/VGG16_100.pth"))

correct = 0
total = 0

model.eval()

with torch.no_grad():
    for i,[image,label] in enumerate(test_loader):
        x = image.to(device)
        y = label.to(device)

        output = model.forward(x)
        _ , output_index = torch.max(output,1)

        total += label.size(0)
        correct += (output_index==y).sum().float()
    
    print("Accuracy of Test Data: {}%".format(100*correct/total))
