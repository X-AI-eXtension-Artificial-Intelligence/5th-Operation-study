import torch.nn as nn
import torch.nn.functional as F

def conv_2_block(input_dim,output_dim):
    model = nn.Sequential(
        nn.Conv2d(input_dim,output_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(output_dim,output_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2) # 2,2 틀림
    )
    return model

def conv_3_block(input_dim,output_dim):
    model = nn.Sequential(
        nn.Conv2d(input_dim,output_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(output_dim,output_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(output_dim,output_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )
    return model

class VGG16(nn.Module):
    def __init__(self, base_dim, num_class=2): #틀림
        super(VGG16,self).__init__() # 틀림
        self.feature = nn.Sequential( # 틀림
            conv_2_block(3,base_dim),
            conv_2_block(base_dim,base_dim*2),
            conv_3_block(base_dim*2,base_dim*4),
            conv_3_block(base_dim*4,base_dim*8),
            conv_3_block(base_dim*8,base_dim*8),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(8*base_dim*7*7,4096),
            nn.ReLU(True),

            nn.Linear(4096,4096),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(4096,1000),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(1000,num_class),
        )
    def forward(self,x):
        x = self.feature(x)
        x = x.view(x.size(0),-1) # 틀림 -> batch만 남기고 푸는 거였음, 뭔가 이상하다 했는데 [32,H,W] -> [32,H*W]
        x = self.fc_layer(x)
        x = F.softmax(x,dim=1)
        return x
