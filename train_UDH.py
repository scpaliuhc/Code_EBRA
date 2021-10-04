from src.model import UDH_Hide,UDH_Reveal,AutoEncoder
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
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def build_models(weight_path,init_hide,init_reveal,enhanced,weight_ae):
    hide_net=UDH_Hide()
    reveal_net=UDH_Reveal()
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
    dataset=Dataset_(data_path,end=num,shuffle=True,scheme='UDH')
    dataloader=DataLoader(dataset,size,shuffle=True,drop_last=True)
    return dataloader

def losses():
    if torch.cuda.is_available():
        # BCE_loss = nn.BCELoss().cuda()
        MSE_loss = nn.MSELoss().cuda()
    else:
        # BCE_loss = nn.BCELoss()
        MSE_loss = nn.MSELoss()
    return MSE_loss

def optims(hide_net,reveal_net,lr=0.001):
    params = list(hide_net.parameters())+list(reveal_net.parameters())
    optimizer = optim.Adam(params, lr=0.001, betas=(0.5, 0.999))
    return optimizer
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 0.001 * (0.1 ** (epoch // 30))
    optimizer.param_groups[0]['lr'] = lr

def main(data_path,weight_path,init_hide,init_reveal,weight_ae,epoches,enhanced,example_path,bs,num_c):
    try:
        os.makedirs(example_path)
    except:
        None

   
    hide_net,reveal_net,ae_net=build_models(weight_path,init_hide,init_reveal,enhanced,weight_ae)
    dataloader_sec=get_dataloader(data_path,bs,4000)
    dataloader_cov=get_dataloader(data_path,bs*num_c,4000)
    train_loader=zip(dataloader_sec,dataloader_cov)
    loader=tqdm(train_loader)
    mse=losses()
    optimizer=optims(hide_net,reveal_net)
    for epoch in range(epoches):
        adjust_learning_rate(optimizer,epoch)
        for idx,(secret,cover) in enumerate(loader):
            if torch.cuda.is_available():
                secret = Variable(secret).cuda()
                cover = Variable(cover).cuda()
            
            secret_nh=secret.repeat(num_c,1,1,1)
            code = hide_net(secret_nh)
            code = code.repeat(num_c,1,1,1)
            output = cover+code
            reveal_secret = reveal_net(output)
            loss_h = mse(output, cover)
            loss_r = mse(reveal_secret, secret_nh)
            if enhanced==True:
                output_T=ae_net(output)
                reveal_secret_T = reveal_net(output_T)
                loss_r = (loss_r + mse(reveal_secret_T, secret_nh))/2

                
            # epoch_loss_h += loss_h.item()
            # epoch_loss_r += loss_r.item()

            loss = loss_h + 0.7 * loss_r
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if idx%300==0:
                if enhanced == True:
                    save_image(torch.cat([cover[:4],output[:4],output_T[:4],secret[:4],reveal_secret[:4],reveal_secret_T[:4]], dim=0), os.path.join(example_path,'epoch_{}.png'.format(epoch)), nrow=4)
                else:
                    save_image(torch.cat([cover[:4],output[:4],secret[:4],reveal_secret[:4]], dim=0), os.path.join(example_path,'epoch_{}.png'.format(epoch)), nrow=4)
        print('=======>>>'*5,'epoch {} is finished'.format(epoch))
        if enhanced:
            hide_net.save(weight_path,'UDHAE_hide_{}.pth'.format(epoch))
            reveal_net.save(weight_path,'UDHAE_reveal_{}.pth'.format(epoch))
        else:
            hide_net.save(weight_path,'UDH_hide_{}.pth'.format(epoch))
            reveal_net.save(weight_path,'UDH_reveal_{}.pth'.format(epoch))

if __name__=='__main__':
    data_path='/data0/lhc/dataset/img_align_celeba/'
    weight_path='checkpoint'
    init_hide=None
    init_reveal=None
    weight_ae='./ae5.pth'
    epoches=200
    enhanced=None
    example_path='./examples/UDH'
    main(data_path,weight_path,init_hide,init_reveal,weight_ae,epoches,enhanced,example_path)
