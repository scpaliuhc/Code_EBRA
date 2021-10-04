import random
from src.model import AutoEncoder,Discriminator,AdversarialLoss,PerceptualLoss,StyleLoss
import torch.nn as nn
import torch.optim as optim
from src.dataset import Dataset_
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
import torch
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def main(data_path,weight_path,epoches,example_path,bs,rb):
    try:
        os.makedirs(example_path)
    except:
        None

    ae_net=AutoEncoder(residual_blocks=rb,in_channels=3).cuda()
    dis_net=Discriminator(in_channels=3,use_sigmoid='nsgan' != 'hinge').cuda()
    l1_loss = nn.L1Loss().cuda()
    perceptual_loss = PerceptualLoss().cuda()
    adversarial_loss = AdversarialLoss(type='nsgan').cuda()
    style_loss = StyleLoss().cuda()
    optim_ae = optim.Adam(
            params=ae_net.parameters(),
            lr=0.0001,
            betas=(0.0, 0.9)
        )

    optim_dis = optim.Adam(
            params=dis_net.parameters(),
            lr=0.0001 * 0.1,
            betas=(0.0, 0.9)
        )

    train_loader = DataLoader(
            dataset=Dataset_(data_path,end=80000,scheme='AE',shuffle=True),
            batch_size=bs,
            drop_last=True,
            shuffle=True
        )
    train_loader=tqdm(train_loader)
    for epoch in range(epoches):
        for idx, images in enumerate(train_loader):
            images=images.cuda()
            outputs=ae_net(images)
            
            gen_loss = 0
            dis_loss = 0
            gen_input_fake = outputs
            gen_fake, _ = dis_net(gen_input_fake)                    # in: [rgb(3)]
            gen_gan_loss = adversarial_loss(gen_fake, True, False) * 0.1
            gen_loss += gen_gan_loss
            gen_l1_loss = l1_loss(outputs, images) * 1
            gen_loss += gen_l1_loss
            gen_content_loss = perceptual_loss(outputs, images)
            gen_content_loss = gen_content_loss * 0.1
            gen_loss += gen_content_loss
            # gen_style_loss = style_loss(outputs * masks, images * masks)
            # gen_style_loss = gen_style_loss * 250
            # gen_loss += gen_style_loss

            optim_ae.zero_grad()
            gen_loss.backward()
            optim_ae.step()

            optim_dis.zero_grad()
            dis_input_real = images
            dis_input_fake = outputs.detach()
            dis_real, _ = dis_net(dis_input_real)                    
            dis_fake, _ = dis_net(dis_input_fake)                    
            dis_real_loss = adversarial_loss(dis_real, True, True)
            dis_fake_loss = adversarial_loss(dis_fake, False, True)
            dis_loss += (dis_real_loss + dis_fake_loss) / 2
            dis_loss.backward()
            optim_dis.step()

            if idx%300==3:
                save_image(torch.cat([images[:4],outputs[:4]]),os.path.join(example_path,'ae_{}_{}.png'.format(epoch,idx)),nrow=4)
        print('=======>>>'*5,'epoch {} is finished'.format(epoch))
        ae_net.save(weight_path,'AE_RB{}_{}.pth'.format(rb,epoch))

# if __name__=='__main__':
#     main()

