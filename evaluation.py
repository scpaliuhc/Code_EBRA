#####################################################################################################
#1. please use train_main.py to train all required models before the evaluation
#2. change the values in config.json to choose different removal attacks and deep hiding schemes
#####################################################################################################
import torch
from src.model import *
import json
from src.load_model import *
from src.dataset import Dataset_
from src.util import *
from torch.utils.data import DataLoader
import datetime
from src.metric import *
import shutil
import numpy as np
from skimage.feature import canny
import piq 
import os
from torchvision.utils import save_image
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def load_config():
    with open('src/config.json','r') as f:
        config=json.load(f)
    return config

def eval_(config):
    PSNR_list=[]
    VIF_list=[]
    FSIM_list=[]
    SSIM_list=[]

    dataset=Dataset_(scheme='ISGAN',shuffle=False,end=2000)
    dataloader=DataLoader(dataset,1,shuffle=False,)
    time=datetime.datetime.now().strftime('%d_%H_%M_%S')
    

    image_dir=os.path.join(config['Image_path'],'{}_{}_{}_{}'.format(
                                                                config['DH']['H_name'].split('_')[0],
                                                                config['PEEL']['scheme'],
                                                                config['Baseline']['scheme'],
                                                                time
    ))
    try:
        os.makedirs(image_dir)
    except:
        print(image_dir,' is existed')
    shutil.copy('src/config.json', '{}/00config.json'.format(image_dir))
    if config['Baseline']['scheme']=="Lattice": 
        lattice=torch.zeros((256,256)).cuda(config['Device'])
        for i in range(0,256,config['Baseline']['size']):
            for j in range(0,256,config['Baseline']['size']):
                lattice[i,j]=1
    
    H,R,Inpainting=load(config)
    if config['Baseline']['scheme']=='FGSM':
       MSE_loss = nn.MSELoss().cuda()
    for idx,(secret,secret_gray,cover) in enumerate(dataloader):
            print('\r {}'.format(idx),end='')
            
            secret=secret.to(config['Device'])
            cover=cover.to(config['Device'])
            secret_gray=secret_gray.to(config['Device'])
            target=cover.clone().detach()

            if config['DH']['scheme']=='UDH':
                stego=H(secret)+cover
            elif config['DH']['scheme']=='DS':
                stego=H(secret,cover)
            elif config['DH']['scheme']=='ISGAN':
                stego=H(secret_gray,cover)   
            stego=torch.clamp(stego,0,1)
            
            ####process
            if config['PEEL']['scheme']!='None':
                gray_stego=stego[0,0,:,:]*0.299+stego[0,1,:,:]*0.587+stego[0,2,:,:]*0.114
                edge_canny=torch.from_numpy(canny(gray_stego.detach().cpu().numpy(), sigma=2).astype(np.float)).float().to(config['Device'])          
                edge_canny=edge_canny.view((1,1,256,256))
                gray_stego=gray_stego.unsqueeze(0)
                if config['PEEL']['scheme']=="PEEL":
                    stego_removal=removal_inpaint_only(stego, Inpainting, edge_canny,s=config['PEEL']['size'],t=config['PEEL']['size'],device=config['Device'])
                elif config['PEEL']['scheme']=="PEELO":
                    stego_removal=removal_DeNoise_part(stego,Inpainting, edge_canny,s=config['PEEL']['size'],t=config['PEEL']['size'],scale=config['PEEL']['scale'],device=config['Device'])
                else:
                    raise NotImplementedError()
            elif config['Baseline']['scheme']!='None':
                if config['Baseline']['scheme']=="GaussianNoise":
                    stego_removal=GaussianNoise(stego, config['Baseline']['scale'], config['Baseline']['channel'])
                elif config['Baseline']['scheme']=="GaussianBlur":
                    stego_removal=GaussianBlur(stego, config['Baseline']['size'], config['Baseline']['sigma'])
                elif config['Baseline']['scheme']=="MedianBlur":   
                    stego_removal=MedianBlur(stego,config['Baseline']['size'])
                elif config['Baseline']['scheme']=="Resize": 
                    stego_removal=Resize(stego) 
                elif config['Baseline']['scheme']=="Lattice": 
                    stego_removal=Lattice(stego,
                    config['Baseline']['scale'],
                    lattice,config['Baseline']['RC'],
                    [config['Baseline']['C1'],config['Baseline']['C2'],config['Baseline']['C3']],
                    config['Device'])   
                elif config['Baseline']['scheme']=="FGSM":
                    stego_removal=FGSM(stego, secret,R,MSE_loss,config['Baseline']['ep'])
                else: 
                    raise ValueError()
                stego_removal=stego_removal.to(config['Device'])


            stego_removal=torch.clamp(stego_removal,0,1)
            reveal_stego=R(stego)
            reveal_stego_removal=R(stego_removal)
            secret_gray=secret_gray.repeat(1,3,1,1)
            if config['DH']['scheme']!='ISGAN':
                psnr=piq.psnr(torch.cat([stego, stego_removal,reveal_stego,reveal_stego_removal]),
                            torch.cat([target,   stego, target,     reveal_stego]),data_range=1.0,reduction='none')
                ssim=piq.ssim(torch.cat([stego, stego_removal,reveal_stego,reveal_stego_removal]),
                            torch.cat([target,   stego, target,     reveal_stego]),data_range=1.0,reduction='none')
                fsim=piq.fsim(torch.cat([stego, stego_removal,reveal_stego,reveal_stego_removal]),
                            torch.cat([target,   stego, target,     reveal_stego]),data_range=1.0,reduction='none')
                vif=piq.vif_p(torch.cat([stego, stego_removal,reveal_stego,reveal_stego_removal]),
                            torch.cat([target,   stego, target,     reveal_stego]),data_range=1.0,reduction='none')
            else:
                psnr=piq.psnr(torch.cat([stego, stego_removal,reveal_stego,reveal_stego_removal]),
                            torch.cat([target,   stego,       secret_gray,      reveal_stego]),data_range=1.0,reduction='none')
                ssim=piq.ssim(torch.cat([stego, stego_removal,reveal_stego,reveal_stego_removal]),
                            torch.cat([target,   stego,       secret_gray,      reveal_stego]),data_range=1.0,reduction='none')
                fsim=piq.fsim(torch.cat([stego, stego_removal,reveal_stego,reveal_stego_removal]),
                            torch.cat([target,   stego,       secret_gray,      reveal_stego]),data_range=1.0,reduction='none')
                vif=piq.vif_p(torch.cat([stego, stego_removal,reveal_stego,reveal_stego_removal]),
                            torch.cat([target,   stego,       secret_gray,      reveal_stego]),data_range=1.0,reduction='none')
            psnr=psnr.cpu().detach().numpy()
            vif=vif.cpu().detach().numpy()
            ssim=ssim.cpu().detach().numpy()
            fsim=fsim.cpu().detach().numpy()
            PSNR_list.append(psnr)
            VIF_list.append(vif)
            SSIM_list.append(ssim)
            FSIM_list.append(fsim)

            save_image(stego,'{}/{}_c_{:.4f}_{:.4f}.jpg'.format(image_dir,idx,psnr[0],vif[0]))
            save_image(reveal_stego,'{}/{}_rc_{:.4f}_{:.4f}.jpg'.format(image_dir,idx,psnr[2],vif[2]))
            save_image(stego_removal,'{}/{}_cp_{:.4f}_{:.4f}.jpg'.format(image_dir,idx,psnr[1],vif[1]))
            save_image(reveal_stego_removal,'{}/{}_rcp_{:.4f}_{:.4f}_{:.4f}.jpg'.format(image_dir,idx,psnr[3],vif[3],ssim[3]))
                    
    PSNR_list=np.array(PSNR_list)
    SSIM_list=np.array(SSIM_list)
    FSIM_list=np.array(FSIM_list)
    VIF_list=np.array(VIF_list)
    with open('{}/00_result.txt'.format(image_dir),'w') as f:
            f.write('PSNR: c\'&c\ts\'&s\tc\"&c\'\ts\"&s\'\n')
            f.write('{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'.format(
                np.mean(PSNR_list[:,0]),
                np.mean(PSNR_list[:,2]),
                np.mean(PSNR_list[:,1]),
                np.mean(PSNR_list[:,3]),
               
            ))
            f.write('SSIM: c\'&c\ts\'&s\tc\"&c\'\ts\"&s\'\n')
            f.write('{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'.format(
                np.mean(SSIM_list[:,0]),
                np.mean(SSIM_list[:,2]),
                np.mean(SSIM_list[:,1]),
                np.mean(SSIM_list[:,3]),

            ))
            f.write('FSIM: c\'&c\ts\'&s\tc\"&c\'\ts\"&s\'\n')
            f.write('{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'.format(
                np.mean(FSIM_list[:,0]),
                np.mean(FSIM_list[:,2]),
                np.mean(FSIM_list[:,1]),
                np.mean(FSIM_list[:,3]),
            ))
            f.write('VIF: c\'&c\ts\'&s\tc\"&c\'\ts\"&s\'\n')
            f.write('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(
                np.mean(VIF_list[:,0]),
                np.mean(VIF_list[:,2]),
                np.mean(VIF_list[:,1]),
                np.mean(VIF_list[:,3]),
            ))

        
if __name__=="__main__":
    config=load_config()
    if config['Baseline']['scheme']!="FGSM":
        with torch.no_grad():
            eval_(config)
    else:
        eval_(config)

    # ###以下是FGSM的代码
    