from src.model import ISGAN_Reveal,ISGAN_Hide,ISGAN_Discriminator,AutoEncoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset import Dataset_
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from torch.autograd import Variable
from torchvision.utils import save_image
# import piq
from src.pytorch_mssim import MSSSIM
from src.ssim import SSIM
import random

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images

        return_images = []
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
            return_images = Variable(torch.cat(return_images, 0))
            return return_images
def build_models(weight_path,init_hide,init_reveal,init_dis,enhanced,weight_ae):
    hide_net=ISGAN_Hide()
    reveal_net=ISGAN_Reveal(ch=1)
    dis_net=ISGAN_Discriminator()
    if enhanced:
        ae_net=AutoEncoder(residual_blocks=5)
        ae_net.load(weight_path,weight_ae)

    if init_hide is not None and init_reveal is not None and init_dis is not None:
        hide_net.load(weight_path,init_hide)
        reveal_net.load(weight_path,init_reveal)
        dis_net.load(weight_path,init_dis)
    else:
        hide_net.init_weight()
        reveal_net.init_weight()

    if torch.cuda.is_available():
        GPU=True
        num=torch.cuda.device_count()
        if num>=2:
            hide_net=nn.DataParallel(hide_net,[0,1])
            reveal_net=nn.DataParallel(reveal_net,[0,1])
            dis_net=nn.DataParallel(dis_net,[0,1])
            if enhanced:
                ae_net=nn.DataParallel(ae_net,[0,1])
        hide_net.cuda()
        reveal_net.cuda()
        dis_net.cuda()
        if enhanced:
            ae_net.cuda()
            ae_net.eval()
    else:
        GPU=False
    if enhanced:
        return hide_net,reveal_net,dis_net,ae_net
    else:
        return hide_net,reveal_net,dis_net,None

def get_dataloader(data_path,size,num):
    dataset=Dataset_(data_path,scheme='ISGAN',end=num,shuffle=True)
    dataloader=DataLoader(dataset,size,shuffle=True)
    return dataloader

def losses():
    if torch.cuda.is_available():
        BCE_loss = nn.BCELoss().cuda()
        MSE_loss = nn.MSELoss().cuda()
        # SSIM_loss=piq.SSIMLoss().cuda()
        # MSSIM_loss = piq.MultiScaleSSIMLoss().cuda() 
        SSIM_loss = SSIM(window_size=11).cuda()
        MSSIM_loss = MSSSIM().cuda()       
    else:
        BCE_loss = nn.BCELoss()
        MSE_loss = nn.MSELoss()
        # SSIM_loss=piq.SSIMLoss()
        # MSSIM_loss = piq.MultiScaleSSIMLoss()  
        SSIM_loss = SSIM(window_size=11).cuda()
        MSSIM_loss = MSSSIM().cuda()  

    return BCE_loss,MSE_loss,SSIM_loss,MSSIM_loss

def optims(hide_net,reveal_net,dis_net,lr=0.0001,lrd=0.0001/3):
    optim_h = torch.optim.Adam(hide_net.parameters(), lr=0.0001,
                                     betas=(0.5, 0.999), 
                                     weight_decay=5e-3)
    optim_r = torch.optim.Adam(reveal_net.parameters(), lr=0.0001,
                                     betas=(0.5, 0.999), 
                                     weight_decay=5e-3)
    optim_d = torch.optim.SGD(dis_net.parameters(), 
                                     lr=lrd, weight_decay=1e-8)

    return optim_h,optim_r,optim_d

def main(data_path,weight_path,init_hide,init_reveal,init_dis,weight_ae,epoches,enhanced,example_path,bs):
    try:
        os.makedirs(example_path)
    except:
        None

    hide_net,reveal_net,dis_net,ae_net=build_models(weight_path,init_hide,init_reveal,init_dis,enhanced,weight_ae)
    dataloader=get_dataloader(data_path,bs,80000)
    loader=tqdm(dataloader)
    BCE_loss,MSE_loss,SSIM_loss,MSSIM_loss=losses()
    optim_h,optim_r,optim_d=optims(hide_net,reveal_net,dis_net)
    pool = ImagePool(50)
    for epoch in range(epoches):
        hide_net.train()
        reveal_net.train()
        if epoch >= 20 and epoch % 20 == 0:
            optim_h.param_groups[0]['lr'] *= 0.9
            optim_r.param_groups[0]['lr'] *= 0.9
            optim_h.param_groups[0]['weight_decay'] *= 0.95
            optim_r.param_groups[0]['weight_decay'] *= 0.95
            
        for idx,(_, secret_gray, cover) in enumerate(loader):
            if torch.cuda.is_available():
                secret = Variable(secret_gray).cuda()
                cover = Variable(cover).cuda()
            target=cover.clone().detach()
            output = hide_net(secret, cover)
            # output=torch.clamp(output,0,1)
            reveal_secret = reveal_net(output)
            # reveal_secret=torch.clamp(reveal_secret,0,1)
            cls1 = dis_net(output)
           
            disloss1 = BCE_loss(cls1, Variable(torch.ones(cls1.size()).cuda() * random.uniform(0.8, 1.2)))
            h_mse = MSE_loss(output, target)
            h_ssim = SSIM_loss(output, target)
            r_mse = MSE_loss(reveal_secret, secret)
            r_ssim = SSIM_loss(reveal_secret, secret)
            if enhanced==True:
                output_T=ae_net(output)
                # output_T=torch.clamp(output_T,0,1)
                reveal_secret_T = reveal_net(output_T)
                # reveal_secret_T=torch.clamp(reveal_secret_T,0,1)
                r_mse=(r_mse+MSE_loss(reveal_secret_T, secret))/2
                r_ssim=(r_ssim+SSIM_loss(reveal_secret_T, secret))/2
            h_loss = (1 - h_ssim) + 0.3 * h_mse
            r_loss = (1 - r_ssim) + 0.3 * h_mse
            loss = h_loss + 0.85 * r_loss + disloss1
            optim_h.zero_grad()
            optim_r.zero_grad()          
            loss.backward()
            optim_h.step()
            optim_r.step()
            if idx%300==0:
                if enhanced == True:
                    save_image(torch.cat([target[:4],output[:4],output_T[:4],secret[:4].repeat(1,3,1,1),reveal_secret[:4].repeat(1,3,1,1),reveal_secret_T[:4].repeat(1,3,1,1)], dim=0), os.path.join(example_path,'epoch_{}.png'.format(epoch)), nrow=4)
                else:
                    save_image(torch.cat([target[:4],output[:4],secret[:4].repeat(1,3,1,1),reveal_secret[:4].repeat(1,3,1,1)], dim=0), os.path.join(example_path,'epoch_{}.png'.format(epoch)), nrow=4)
            if idx % 5 == 0:
                cls_cover = dis_net(target)
                stego = pool.query(output)
                cls_stego = dis_net(stego)
                discover = BCE_loss(cls_cover, Variable(torch.ones(cls_cover.size()).cuda() * random.uniform(0.8, 1.2)))
                disstego = BCE_loss(cls_stego, Variable(torch.ones(cls_stego.size()).cuda() * random.uniform(0., 0.2)))
                disloss = discover + disstego
                optim_d.zero_grad()
                disloss.backward()
                optim_d.step()
        print('=======>>>'*5,'epoch {} is finished'.format(epoch))
        # if enhanced:
        #     hide_net.save(weight_path,'ISGANAE_hide_{}.pth'.format(epoch))
        #     reveal_net.save(weight_path,'ISGANAE_reveal_{}.pth'.format(epoch))
        #     dis_net.save(weight_path,'ISGANAE_dis_{}.pth'.format(epoch))
        # else:
        #     hide_net.save(weight_path,'ISGAN_hide_{}.pth'.format(epoch))
        #     reveal_net.save(weight_path,'ISGAN_reveal_{}.pth'.format(epoch))
        #     dis_net.save(weight_path,'ISGAN_dis_{}.pth'.format(epoch))
        

# if __name__=='__main__':
#     data_path='/data0/lhc/dataset/img_align_celeba/'
#     weight_path='checkpoint/ISGAN_training'
#     init_hide=None
#     init_reveal=None
#     init_dis=None
#     weight_ae='./AE_RB5.pth'
#     epoches=200
#     enhanced=None
#     example_path='./examples/ISGAN'
#     main(data_path,weight_path,init_hide,init_reveal,init_dis,weight_ae,epoches,enhanced,example_path)
