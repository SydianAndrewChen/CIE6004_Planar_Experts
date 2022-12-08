import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):#groups=1普通卷积
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.ReLU6(inplace=True)
        )

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(out_ch, out_ch, 5, padding="same")
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 7, padding="same")
        # nn.init.xavier_uniform_(self.conv1.weight)
        # nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class Encoder(nn.Module):
    def __init__(self, chs=(3,64,128,256,512,1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i], 2, 2) for i in range(len(chs)-1)])
        # for upconv in self.upconvs:
        #     nn.init.xavier_uniform_(upconv.weight)
        self.dec_blocks = nn.ModuleList([ConvBNReLU(chs[i], chs[i+1], 5) for i in range(len(chs)-1)]) 
        
        # self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
    def forward(self, x):
        # for i in range(len(self.chs)-1):
        #     x = self.upconvs[i](x)
        #     x = self.dec_blocks[i](x)
        #     x = F.relu(x)
        x = self.dec_blocks[0](x)
        x = self.upconvs[0](x)
        x = self.dec_blocks[1](x)
        x = F.interpolate(x, size=[512, 512], mode="bilinear")
        # print(x.shape)
        # for i in range(len(self.chs)-1):
        #     x        = self.upconvs[i](x)
            # print(f"x.shape = \n{x.shape}\n")
            # print(f"self.dec_blocks[i] = \n{self.dec_blocks[i]}\n")
            # exit()
        return x
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class UNet(nn.Module):
    # def __init__(self, enc_chs=(3,64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64), num_class=1, retain_dim=False, out_sz=(572,572)):
    def __init__(self, enc_chs=(3,64,128,256), dec_chs=(256, 128, 64), num_class=3, retain_dim=False, out_sz=(128,128)):
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim  = retain_dim
        self.out_sz = out_sz

    def forward(self, x):
        print('x:',x.shape)
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)
        return out

if __name__ == '__main__':
    # unet = UNet()
    # output = unet(input_)
    input_ = torch.ones(1, 32, 512, 512)
    decoder = Decoder(chs=(32, 16, 8, 3))
    output = decoder(input_)
    print(f"input_.shape = \n{input_.shape}\n")
    print(f"output.shape = \n{output.shape}\n")