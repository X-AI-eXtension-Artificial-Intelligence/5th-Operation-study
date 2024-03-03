import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        ## 네트워크에서 반복적으로 사용되는 Conv + batchNorm + ReLU를 합쳐서 함수로 정의
        # 커널 사이즈가 3x3인 Conv layer
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                          kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU()
                )
            return layers 
        
        # ---------constracting path (Encoder) -----------------
        # output : [572, 572, 1(or 3)] -> [570, 570, 64] 실제로 이미지 크기가 줄어드는 것이 아님!! 이미지 크기가 그대로긴 함
         # (인풋 - 피처맵 + 2*패딩)/stride +1  
        self.enc1_1 = CBR2d(in_channels=3, out_channels=64)  #해당 이미지가 컬러일 경우 3, 64
        # output : [570, 570, 64] -> [568, 568, 64]
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)

        # pooling == downSampling(절반)
        # output : [568, 568, 64] -> [284, 284, 64]
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # output : [284, 284, 64] -> [282, 282,128]
        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        # output : [282, 282,128] -> [280, 280,128]
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)
        
        # output : [280, 280, 128] -> [140, 140,128]
        self.pool2 = nn.MaxPool2d(2)

        # output : 140, 140,128]-> [138, 138, 256]
        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        # output :  [138, 138, 256] -> [136, 136, 256]
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)

        # output :  [136, 136, 256] -> [68, 68, 256]
        self.pool3 = nn.MaxPool2d(2)

        # output :  [68, 68, 256] -> [66, 66, 512]
        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        # output :  [66, 66, 512] -> [64, 64, 512]
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)

        # output :  [64, 64, 512] -> [32, 32, 512]
        self.pool4 = nn.MaxPool2d(2)

        # output :  [32, 32, 512] -> [30, 30, 1024]
        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)
        # output :  [30, 30, 1024] -> [28, 28, 1024]
        self.enc5_2 = CBR2d(in_channels=1024, out_channels=1024)


        # --------- expansive path (Decoder)--------------------
        # output :  [28, 28, 1024] -> [56, 56, 512]  ??? 512로 줄어드는지? -> test해보기  -> 아닌듯..?
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        # output :  [56, 56, 512] -> [54, 54, 512]
        self.dec1_1 = CBR2d(in_channels=1024, out_channels=512)
        # output :  [54, 54, 512] -> [52, 52, 512]
        self.dec1_2 = CBR2d(in_channels=512, out_channels=512)   # 512??? 1024???

        # output :  [52, 52, 512] -> [104, 104, 256]
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        # output :  [104, 104, 256] -> [102, 102, 256]
        self.dec2_1 = CBR2d(in_channels=512, out_channels=256)
        # output :  [102, 102, 256] -> [100, 100, 256]
        self.dec2_2 = CBR2d(in_channels=256, out_channels=256)

        # output :  [100, 100, 256] -> [200, 200, 128]
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # output :  [200, 200, 128] -> [188, 188, 128]
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)
        # output :  [188, 188, 128] -> [186, 186, 128]
        self.dec3_2 = CBR2d(in_channels=128, out_channels=128)

        # output :  [186, 186, 128] -> [392, 392, 64]
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # output :  [392, 392, 64] -> [390, 390, 64]
        self.dec4_1 = CBR2d(in_channels=128, out_channels=64)
        # output :  [390, 390, 64] -> [388, 388, 64]
        self.dec4_2 = CBR2d(in_channels=64, out_channels=64)

        # output :  [388, 388, 64] -> [388, 388, 2]
        self.outconv = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)  # RGB이면 3?


    def forward(self, x):
        # Encoder
        e1_1 = self.enc1_1(x)
        e1_2 = self.enc1_2(e1_1)
        e1_p = self.pool1(e1_2)

        e2_1 = self.enc2_1(e1_p)
        e2_2 = self.enc2_2(e2_1)
        e2_p = self.pool2(e2_2)

        e3_1 = self.enc3_1(e2_p)
        e3_2 = self.enc3_2(e3_1)
        e3_p = self.pool3(e3_2)

        e4_1 = self.enc4_1(e3_p)
        e4_2 = self.enc4_2(e4_1)
        e4_p = self.pool4(e4_2)

        e5_1 = self.enc5_1(e4_p)
        e5_2 = self.enc5_2(e5_1)

        # Decoder
        d1_up = self.upconv1(e5_2)
        #### skip connection
        d1_c = torch.cat([d1_up,e4_2], dim=1)
        d1_1 = self.dec1_1(d1_c)
        d1_2 = self.dec1_2(d1_1)

        d2_up = self.upconv2(d1_2)
        d2_c = torch.cat([d2_up, e3_2], dim=1)
        d2_1 = self.dec2_1(d2_c)
        d2_2 = self.dec2_2(d2_1)

        d3_up = self.upconv3(d2_2)
        # crop 과정이 빠진 것!!  cvl2d의 padding을 0으로 바꾸고 crop해서 사이즈 맞추는 과정 추가!!
        d3_c = torch.cat([d3_up, e2_2], dim=1)
        d3_1 = self.dec3_1(d3_c)
        d3_2 = self.dec3_2(d3_1)

        d4_up = self.upconv4(d3_2)
        d4_c = torch.cat([d4_up, e1_2], dim=1)
        d4_1 = self.dec4_1(d4_c)
        d4_2 = self.dec4_2(d4_1)

        out = self.outconv(d4_2)
        out = out.view(out.size(0), -1)

        return out