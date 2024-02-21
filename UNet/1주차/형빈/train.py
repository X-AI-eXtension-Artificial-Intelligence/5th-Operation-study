import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms,datasets

from unet import UNet

class Dataset(torch.utils.data.Dataset):
    def __init__(self,data_dir,transform = None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir)

        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]

        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)
    
    def __getitem__(self,index):
        label = np.load(os.path.join(self.data_dir,self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        # normalize
        label = label/255
        input = input/255

        # if black white image (HW) -> make 3rd channel (HWC)

        if label.ndim ==2:
            label = label[:,:,np.newaxis]

        if input.ndim == 2:
            input = input[:,:,np.newaxis]

        data = {'input':input, 'label':label}

        # if transform -> road transform

        if self.transform:
            data = self.transform(data)

        return data


#################################
# Transformation

class ToTensor(object):
    def __call__(self,data):
        label,input = data['label'], data['input']
        #print(label.shape,input,shape)

        label = label.transpose((2,0,1)).astype(np.float32) #torch needs CHW
        input = input.transpose((2,0,1)).astype(np.float32)
        
        # == torch.tensor()
        # from_numpy is share memory location with past numpy array and present tensor
        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data
    
class Normalization(object):
    def __init__(self,mean=0.5,std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self,data):
        label, input = data['label'], data['input']

        input = (input-self.mean) / self.std

        data = {'label':label,'input':input}

        return data
    
class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data

####################################
# save networks
    
def save(ckpt_dir,net,optim,epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net':net.state_dict(),'optim':optim.state_dict()},
               "%s/model_epoch%d.pth"%(ckpt_dir,epoch))
    
# load network
    
def load(ckpt_dir,net,optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net,optim,epoch
    
    # os.listdir :  return list of files or dirs in the directory
    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int("".join(filter(str.isdigit,f))))

    dict_model = torch.load("%s/%s"%(ckpt_dir,ckpt_lst[-1]))

    ## torch.nn.Module.load_state_dict -> https://tutorials.pytorch.kr/beginner/saving_loading_models.html
    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net,optim,epoch

###################################
# set train parameters

lr = 1e-3
batch_size = 4
num_epoch = 20

base_dir = '/Users/hyungson/vscode/xai/unet'
dir_data = '/Users/hyungson/vscode/xai/unet/dataset'

data_dir = dir_data
ckpt_dir = os.path.join(base_dir, "checkpoint")
log_dir = os.path.join(base_dir,"log")

# transform and dataloader for training
transform = transforms.Compose([Normalization(mean=0.5,std=0.5),RandomFlip(),ToTensor()])

dataset_train = Dataset(data_dir = os.path.join(data_dir,'train'),transform = transform)
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,num_workers=0)

dataset_val = Dataset(data_dir = os.path.join(data_dir,'val'),transform=transform)
loader_val = DataLoader(dataset_val,batch_size=batch_size,shuffle=False,num_workers=1)

# create network
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = UNet().to(device)

# loss func, optim
fn_loss = nn.BCEWithLogitsLoss().to(device)
optim = torch.optim.Adam(net.parameters(),lr=lr)

# etc variables, functions
num_data_train = len(dataset_train)
num_data_val = len(dataset_val)

num_batch_train = np.ceil(num_data_train / batch_size) # np.ceil 올림 함수
num_batch_val = np.ceil(num_data_val / batch_size)

fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0,2,3,1)
fn_denorm = lambda x,mean, std : (x*std) + mean
fn_class = lambda x: 1.0 * (x>0.5)

# SummaryWriter for Tensorboard
writer_train = SummaryWriter(log_dir=os.path.join(log_dir,'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir,'val'))



#############################
# train network
st_epoch = 0
# load model if trained model existed
net,optim,st_epoch = load(ckpt_dir=ckpt_dir,net=net,optim=optim)
def train():
    for epoch in range(st_epoch + 1, num_epoch + 1):
        print(epoch,"epoch processing....")
        net.train()
        loss_arr = []

        # start index at 1

        for batch,data in enumerate(loader_train,1):
            # forward pass
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            # backward pass
            optim.zero_grad()

            loss = fn_loss(output,label)
            loss.backward()

            optim.step()

            # calculate loss_fn
            loss_arr += [loss.item()]

            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                (epoch,num_epoch,batch,num_batch_train,np.mean(loss_arr)))
            
            # save at Tensorboard

            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input, mean=0.5,std=0.5))
            output = fn_tonumpy(fn_class(output))

            writer_train.add_image('label',label,num_batch_train*(epoch-1)+batch, dataformats='NHWC')
            writer_train.add_image('input',input,num_batch_train*(epoch-1)+batch, dataformats='NHWC')
            writer_train.add_image('output',output,num_batch_train*(epoch-1)+batch, dataformats='NHWC')

        writer_train.add_scalar('loss',np.mean(loss_arr),epoch)

        with torch.no_grad():
            net.eval()
            loss_arr = []

            for batch,data in enumerate(loader_val,1):
                #forward pass
                label = data['label'].to(device)
                input = data['input'].to(device)

                output = net(input)

                # calculate loss fn
                loss = fn_loss(output,label)
                loss_arr += [loss.item()]

                print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                    (epoch,num_epoch,batch,num_batch_val,np.mean(loss_arr)))
                

                # save at Tensorboard
                label = fn_tonumpy(label)
                input = fn_tonumpy(fn_denorm(input,mean=0.5,std=0.5))
                output= fn_tonumpy(fn_class(output))

                writer_val.add_image('label',label,num_batch_val * (epoch-1) + batch, dataformats='NHWC')
                writer_val.add_image('input',input,num_batch_val * (epoch-1) + batch, dataformats='NHWC')
                writer_val.add_image('output',output,num_batch_val * (epoch-1) + batch, dataformats='NHWC')

            writer_val.add_scalar('loss',np.mean(loss_arr),epoch)

            # model save per epoch 50
            if epoch % 50 == 0:
                save(ckpt_dir=ckpt_dir,net=net,optim=optim,epoch=epoch)

            writer_train.close()
            writer_val.close()


if __name__ == '__main__':
    train()