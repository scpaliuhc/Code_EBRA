from src.model import DS_Hide,DS_Reveal,AutoEncoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset import Dataset_
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from torch.autograd import Variable
from torchvision.utils import save_image
import os

def build_models(weight_path,init_hide,init_reveal,enhanced,weight_ae):
    hide_net=DS_Hide()
    reveal_net=DS_Reveal()
    if enhanced:
        ae_net=AutoEncoder(residual_blocks=5)
        ae_net.load(weight_path,weight_ae)

    if init_hide is not None and init_reveal is not None:
        hide_net.load(weight_path,init_hide)
        reveal_net.load(weight_path,init_reveal)
    else:
        hide_net.init_weight()
        reveal_net.init_weight()

    if torch.cuda.is_available():
        GPU=True
        num=torch.cuda.device_count()
        if num>=2:
            hide_net=nn.DataParallel(hide_net,[0,1])
            reveal_net=nn.DataParallel(reveal_net,[0,1])
            if enhanced:
                ae_net=nn.DataParallel(ae_net,[0,1])
        hide_net.cuda()
        reveal_net.cuda()
        if enhanced:
            ae_net.cuda()
            ae_net.eval()
    else:
        GPU=False
    if enhanced:
        return hide_net,reveal_net,ae_net
    else:
        return hide_net,reveal_net,None

def get_dataloader(data_path,size,num):
    dataset=Dataset_(data_path,end=num,shuffle=True)
    dataloader=DataLoader(dataset,size,shuffle=True)
    return dataloader

def losses():
    if torch.cuda.is_available():
        MSE_loss = nn.MSELoss().cuda()
    else:
        MSE_loss = nn.MSELoss()
    return MSE_loss

def optims(hide_net,reveal_net,lr=1e-3):
    optim_h = optim.Adam(hide_net.parameters(), lr=lr)
    optim_r = optim.Adam(reveal_net.parameters(), lr=lr)
    schedulee_h = MultiStepLR(optim_h, milestones=[100, 1000])
    schedulee_r = MultiStepLR(optim_h, milestones=[100, 1000])
    return optim_h,optim_r,schedulee_h,schedulee_r

def main(data_path,weight_path,init_hide,init_reveal,weight_ae,epoches,enhanced,example_path,bs):
    try:
        os.makedirs(example_path)
    except:
        None

    hide_net,reveal_net,ae_net=build_models(weight_path,init_hide,init_reveal,enhanced,weight_ae)
    dataloader=get_dataloader(data_path,bs,80000)
    loader=tqdm(dataloader)
    mse=losses()
    optim_h,optim_r,schedulee_h,schedulee_r=optims(hide_net,reveal_net)
    for epoch in range(epoches):
        schedulee_h.step()
        schedulee_r.step()
        # epoch_loss_h = 0.
        # epoch_loss_r = 0.
        for idx,(secret,cover) in enumerate(loader):
            if torch.cuda.is_available():
                secret = Variable(secret).cuda()
                cover = Variable(cover).cuda()
            output = hide_net(secret, cover)
            reveal_secret = reveal_net(output)
            loss_h = mse(output, cover)
            loss_r = mse(reveal_secret, secret)
            if enhanced==True:
                output_T=ae_net(output)
                reveal_secret_T = reveal_net(output_T)
                loss_r = (loss_r + mse(reveal_secret_T, secret))/2

            loss = loss_h + 0.7 * loss_r
            optim_h.zero_grad()
            optim_r.zero_grad()
            loss.backward()
            optim_h.step()
            optim_r.step()
            if idx%300==0:
                if enhanced == True:
                    save_image(torch.cat([cover[:4],output[:4],output_T[:4],secret[:4],reveal_secret[:4],reveal_secret_T[:4]], dim=0), os.path.join(example_path,'epoch_{}.png'.format(epoch)), nrow=4)
                else:
                    save_image(torch.cat([cover[:4],output[:4],secret[:4],reveal_secret[:4]], dim=0), os.path.join(example_path,'epoch_{}.png'.format(epoch)), nrow=4)
        print('=======>>>'*5,'epoch {} is finished'.format(epoch))
        if enhanced:
            hide_net.save(weight_path,'DSAE_hide_{}.pth'.format(epoch))
            reveal_net.save(weight_path,'DSAE_reveal_{}.pth'.format(epoch))
        else:
            hide_net.save(weight_path,'DS_hide_{}.pth'.format(epoch))
            reveal_net.save(weight_path,'DS_reveal_{}.pth'.format(epoch))

# if __name__=='__main__':
#     data_path='/data0/lhc/dataset/img_align_celeba/'
#     weight_path='checkpoint/DS_training'
#     init_hide=None
#     init_reveal=None
#     weight_ae='./AE_RB5.pth'
#     epoches=200
#     enhanced=False
#     example_path='./examples/DS'
#     main(data_path,weight_path,init_hide,init_reveal,weight_ae,epoches,enhanced,example_path)
