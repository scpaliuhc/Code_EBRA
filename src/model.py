from ntpath import join
import torch
import torch.nn as nn
import os
import functools
from src.util import load_model_for_par
# from src.dataset import Dataset_
# from torch.utils.data import DataLoader
# from torchvision.utils import save_image
import torch.nn.functional as F
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1)

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
    def load(self,path,name):
        state_dict=load_model_for_par(os.path.join(path,name))
        self.load_state_dict(state_dict)
        print(os.path,join(path,name),' is loaded')
    def save(self,path,name):
        try:
            torch.save(self.module.state_dict(),os.path,join(path,name))
        except:
            torch.save(self.module.state_dict(),os.path,join(path,name))
        finally:
            print(os.path,join(path,name),' is saved')

    

##############DS##################
class DS_Hide(BaseModel):
    def __init__(self):
        super(DS_Hide, self).__init__()
        self.prepare = nn.Sequential(
            conv3x3(3, 64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(True)
        )
        self.hidding_1 = nn.Sequential(
            nn.Conv2d(128, 64, 1, 1, 0),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            conv3x3(64, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            conv3x3(64, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

        self.hidding_2 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            conv3x3(32, 16),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            conv3x3(16, 3),
            nn.Tanh()
        )
    def forward(self, secret, cover):
        secret=(secret-0.5)*2 #0,1 to -1,1
        cover=(cover-0.5)*2
        sec_feature = self.prepare(secret)
        cover_feature = self.prepare(cover)
        out = self.hidding_1(torch.cat([sec_feature, cover_feature], dim=1))
        out = self.hidding_2(out)
        out = out/2+0.5 #-1,1 to 0,1
        return out
    def init_weight(self):
        def init_core(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.xavier_uniform(m.weight)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
            elif classname.find('ConvTranspose2d') != -1:
                nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0)
        self.apply(init_core)

class DS_Reveal(BaseModel):
    def __init__(self):
        super(DS_Reveal, self).__init__()
        self.reveal = nn.Sequential(
            conv3x3(3, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            conv3x3(32, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            conv3x3(32, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            conv3x3(64, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            conv3x3(32, 16),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            conv3x3(16, 3),
            nn.Tanh()
        )
    def forward(self, image):
        image=(image-0.5)*2
        out = self.reveal(image)
        out=out/2+0.5
        return out
    def init_weight(self):
        def init_core(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.xavier_uniform(m.weight)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
            elif classname.find('ConvTranspose2d') != -1:
                nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0)
        self.apply(init_core)



##############ISGAN################
class InceptionModule(nn.Module):
    def __init__(self, in_nc, out_nc, bn='BatchNorm'):
        super(InceptionModule, self).__init__()
        self.conv1x1 = nn.Conv2d(in_nc, out_nc//4, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm1x1 = nn.BatchNorm2d(out_nc//4)

        self.conv1x1_2 = nn.Conv2d(in_nc, out_nc//4, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3x3 = nn.Conv2d(out_nc//4, out_nc//4, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm3x3 = nn.BatchNorm2d(out_nc//4)  # nn.Sequential()  #

        self.conv1x1_3 = nn.Conv2d(in_nc, out_nc//4, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv5x5 = nn.Conv2d(out_nc//4, out_nc//4, kernel_size=5, stride=1, padding=2, bias=False)
        self.norm5x5 = nn.BatchNorm2d(out_nc//4)

        self.maxpooling = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv1x1_4 = nn.Conv2d(in_nc, out_nc//4, kernel_size=1, stride=1, padding=0, bias=False)
        self.normpooling = nn.BatchNorm2d(out_nc//4)

        self.conv1x1_5 = nn.Conv2d(in_nc, out_nc, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.LeakyReLU()  # nn.ReLU(True)

    def forward(self, x):
        out1x1 = self.relu(self.norm1x1(self.conv1x1(x)))
        out3x3 = self.relu(self.conv1x1_2(x))
        out3x3 = self.relu(self.norm3x3(self.conv3x3(out3x3)))
        out5x5 = self.relu(self.conv1x1_3(x))
        out5x5 = self.relu(self.norm5x5(self.conv5x5(out5x5)))
        outmaxpooling = self.maxpooling(x)
        outmaxpooling = self.relu(self.norm5x5(self.conv1x1_4(outmaxpooling)))

        out = torch.cat([out1x1, out3x3, out5x5, outmaxpooling], dim=1)
        residual = self.conv1x1_5(x)
        out = out + residual
        return out

class ISGAN_Hide(BaseModel):
    def __init__(self, in_nc=2, out_nc=1):
        super(ISGAN_Hide, self).__init__()
        self.conv1 = nn.Conv2d(in_nc, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(16)
        self.block1 = InceptionModule(in_nc=16, out_nc=32)
        self.block2 = InceptionModule(in_nc=32, out_nc=64)
        self.block3 = InceptionModule(in_nc=64, out_nc=128)
        self.block7 = InceptionModule(in_nc=128, out_nc=256)
        self.block8 = InceptionModule(in_nc=256, out_nc=128)
        self.block4 = InceptionModule(in_nc=128, out_nc=64)
        self.block5 = InceptionModule(in_nc=64, out_nc=32)
        self.block6 = InceptionModule(in_nc=32, out_nc=16)
        self.conv2 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(3)
        self.conv3 = nn.Conv2d(3, out_nc, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.LeakyReLU()  # nn.ReLU(True)
        self.tanh = nn.Tanh()

    def forward(self, secret, cover):
        # convert cover from rgb to yuv
        Y = 0 + 0.299 * cover[:, 0, :, :] + 0.587 * cover[:, 1, :, :] + 0.114 * cover[:, 2, :, :]
        CB = 128.0/255 - 0.168736 * cover[:, 0, :, :] - 0.331264 * cover[:, 1, :, :] + 0.5 * cover[:, 2, :, :]
        CR = 128.0/255 + 0.5 * cover[:, 0, :, :] - 0.418688 * cover[:, 1, :, :] - 0.081312 * cover[:, 2, :, :]
        Y = Y.unsqueeze(1)
        x = torch.cat([secret, Y], dim=1)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block7(out)
        out = self.block8(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.relu(self.norm2(self.conv2(out)))
        out = self.tanh(self.conv3(out))
        cover[:, 0, :, :] = out[:, 0, :, :] + 1.402 * CR - 1.402 * 128.0/255
        cover[:, 1, :, :] = out[:, 0, :, :] - 0.344136 * CB + 0.344136 * 128.0/255 - 0.714136 * CR + 0.714136 * 128.0/255
        cover[:, 2, :, :] = out[:, 0, :, :] + 1.772 * CB - 1.772 * 128.0/255
        return cover

    def init_weight(self):
        def init_core(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.xavier_uniform(m.weight)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
            elif classname.find('ConvTranspose2d') != -1:
                nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0)
        self.apply(init_core)

class ISGAN_Reveal(BaseModel):
    def __init__(self, nc=1, nhf=32, output_function=nn.Sigmoid,ch=3):
        super(ISGAN_Reveal, self).__init__()
        self.ch=ch
        self.main = nn.Sequential(
            nn.Conv2d(nc, nhf, 3, 1, 1),
            nn.BatchNorm2d(nhf),
            nn.ReLU(True),
            nn.Conv2d(nhf, nhf * 2, 3, 1, 1),
            nn.BatchNorm2d(nhf * 2),
            nn.ReLU(True),
            nn.Conv2d(nhf * 2, nhf * 4, 3, 1, 1),
            nn.BatchNorm2d(nhf * 4),
            nn.ReLU(True),
            nn.Conv2d(nhf * 4, nhf * 2, 3, 1, 1),
            nn.BatchNorm2d(nhf * 2),
            nn.ReLU(True),
            nn.Conv2d(nhf * 2, nhf, 3, 1, 1),
            nn.BatchNorm2d(nhf),
            nn.ReLU(True),
            nn.Conv2d(nhf, nc, 3, 1, 1),
            output_function())

    def forward(self, stego):
        Y = 0 + 0.299 * stego[:, 0, :, :] + 0.587 * stego[:, 1, :, :] + 0.114 * stego[:, 2, :, :]
        Y = Y.unsqueeze(1)
        output = self.main(Y)
        if self.ch==3:
            output = output.repeat(1,3,1,1)
        return output
    def init_weight(self):
        def init_core(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.xavier_uniform(m.weight)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
            elif classname.find('ConvTranspose2d') != -1:
                nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0)
        self.apply(init_core)

class ISGAN_Discriminator(BaseModel):
    def __init__(self, kernel_size=5, padding=2):
        super(ISGAN_Discriminator, self).__init__()
        # self.preprocessing = kv_conv2d()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.AvgPool2d(kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d(5, 2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.AvgPool2d(5, 2)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool4 = nn.AvgPool2d(kernel_size=5, stride=2)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        # self.pool5 = nn.AvgPool2d(kernel_size=16, stride=1)
        # self.pool6 = nn.AvgPool2d(kernel_size=4, stride=4)
        # self.pool7 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(128*(1+2*2+4*4), 128)
        self.fc2 = nn.Linear(128, 2)
        self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = x  # self.preprocessing(x)
        out = self.conv1(out)
        out = torch.abs(out)
        # print(out)
        out = self.relu(self.bn1(out))
        out = self.pool1(out)
        out = self.tanh(self.bn2(self.conv2(out)))
        out = self.pool2(out)
        out = self.relu(self.bn3(self.conv3(out)))
        out = self.pool3(out)
        out = self.relu(self.bn4(self.conv4(out)))
        out = self.pool4(out)
        out = self.relu(self.bn5(self.conv5(out)))
        _, _, x, y = out.size()
        out1 = F.avg_pool2d(out, kernel_size=(x//4, y//4), stride=(x//4, y//4))
        out2 = F.avg_pool2d(out, kernel_size=(x//2, y//2), stride=(x//2, y//2))
        out3 = F.avg_pool2d(out, kernel_size=(x, y), stride=1)
        # print(out.size(),out1.size(),out2.size(),out3.size())
        out1 = out1.view(out1.size(0), -1)
        out2 = out2.view(out2.size(0), -1)
        out3 = out3.view(out3.size(0), -1)
        out = torch.cat([out1, out2, out3], dim=1)
        # print(out.size())
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return self.sigmoid(out)
##############UDH################
class UDH_Hide(BaseModel):
    def __init__(self, input_nc=3, output_nc=3, num_downs=5, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, output_function=nn.Tanh):
        super(UDH_Hide, self).__init__()   
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, output_function=output_function)

        self.model = unet_block
        self.tanh = output_function==nn.Tanh
        if self.tanh:
            self.factor = 10/255
        else:
            self.factor = 1.0

    def forward(self, input):
        return self.factor*self.model(input)
    
    def init_weight(self):
        def init_core(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                import torch.nn.init as init
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
            elif classname.find('BatchNorm') != -1:
                m.weight.data.fill_(1.0) 
                m.bias.data.fill_(0)
        self.apply(init_core)

class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,submodule=None, outermost=False, innermost=False, norm_layer=None, use_dropout=False, output_function=nn.Sigmoid):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if norm_layer == None:
            use_bias = True
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        uprelu = nn.ReLU(True)
        if norm_layer != None:
            downnorm = norm_layer(inner_nc)
            upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            if output_function == nn.Tanh:
                up = [uprelu, upconv, nn.Tanh()]
            else:
                up = [uprelu, upconv, nn.Sigmoid()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            if norm_layer == None:
                up = [uprelu, upconv]
            else:
                up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            if norm_layer == None:
                down = [downrelu, downconv]
                up = [uprelu, upconv]
            else: 
                down = [downrelu, downconv, downnorm]
                up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

class UDH_Reveal(BaseModel):
    def __init__(self, input_nc=3, output_nc=3, nhf=64, norm_layer=nn.BatchNorm2d, output_function=nn.Sigmoid):
        super(UDH_Reveal, self).__init__()
        # input is (3) x 256 x 256

        self.conv1 = nn.Conv2d(input_nc, nhf, 3, 1, 1)        
        self.conv2 = nn.Conv2d(nhf, nhf * 2, 3, 1, 1)
        self.conv3 = nn.Conv2d(nhf * 2, nhf * 4, 3, 1, 1)
        self.conv4 = nn.Conv2d(nhf * 4, nhf * 2, 3, 1, 1)
        self.conv5 = nn.Conv2d(nhf * 2, nhf, 3, 1, 1)
        self.conv6 = nn.Conv2d(nhf, output_nc, 3, 1, 1)
        self.output=output_function()
        self.relu = nn.ReLU(True)

        self.norm_layer = norm_layer
        if norm_layer != None:
            self.norm1 = norm_layer(nhf)
            self.norm2 = norm_layer(nhf*2)
            self.norm3 = norm_layer(nhf*4)
            self.norm4 = norm_layer(nhf*2)
            self.norm5 = norm_layer(nhf)

    def forward(self, input):

        if self.norm_layer != None:
            x=self.relu(self.norm1(self.conv1(input)))
            x=self.relu(self.norm2(self.conv2(x)))
            x=self.relu(self.norm3(self.conv3(x)))
            x=self.relu(self.norm4(self.conv4(x)))
            x=self.relu(self.norm5(self.conv5(x)))
            x=self.output(self.conv6(x))
        else:
            x=self.relu(self.conv1(input))
            x=self.relu(self.conv2(x))
            x=self.relu(self.conv3(x))
            x=self.relu(self.conv4(x))
            x=self.relu(self.conv5(x))
            x=self.output(self.conv6(x))

        return x
    def init_weight(self):
        def init_core(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                import torch.nn.init as init
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
            elif classname.find('BatchNorm') != -1:
                m.weight.data.fill_(1.0) 
                m.bias.data.fill_(0)
        self.apply(init_core)

##############Inpainting & Auto-encoder###########
def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module

class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]

class AutoEncoder(BaseModel):
    def __init__(self, residual_blocks=8,in_channels=3, init_weights=True):
        super(AutoEncoder, self).__init__()
        self.residual_blocks=residual_blocks
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = (torch.tanh(x) + 1) / 2
        return x

class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        import torchvision.models as models
        features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out

class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss

class StyleLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

        return style_loss

class PerceptualLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])


        return content_loss

# def Remove_zip_and_generator_key():
