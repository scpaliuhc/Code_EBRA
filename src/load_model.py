from src.model import *
import os
def load_DS(path,name1,name2,device='cuda:0'):
    H=DS_Hide()
    R=DS_Reveal()
    H.load(path,name1)
    R.load(path,name2)
    R.eval()
    H.eval()
    R.to(device)
    H.to(device)
    return H,R
def load_ISGAN(path,name1,name2,device='cuda:0'):
    H=ISGAN_Hide()
    R=ISGAN_Reveal()
    H.load(path,name1)
    R.load(path,name2)
    # H=H.eval()
    R=R.eval()
    # H.eval()
    R.cuda(device)
    H.cuda(device)
    return H,R
def load_UDH(path,name1,name2,device='cuda:0'):
    H=UDH_Hide()
    R=UDH_Reveal()
    H.load(path,name1)
    R.load(path,name2)
    R.eval()
    H.eval()
    R.cuda(device)
    H.cuda(device)
    return H,R

def load_Stego(scheme,path,name1,name2,device='cuda:0'):
    if scheme=='DS':
        return load_DS(path, name1, name2,device=device)
    elif scheme=='ISGAN':
        return load_ISGAN(path, name1, name2,device=device)
    elif scheme=='UDH':
        return load_UDH(path, name1, name2,device=device)
    else:
        raise NotImplementedError()

def load_AE(rb,path,name,device='cuda:0',inch=3):
    AE=AutoEncoder(residual_blocks=rb,in_channels=inch)
    AE.load(path,name)
    AE.eval()
    AE.cuda(device)
    return AE

# def load_Blur(path,name,device='cuda:0'):
#     Blur=EdgeConnBlur()
#     Blur.load(path,name)
#     Blur.eval()
#     Blur.cuda(device)
#     return Blur

def load_model_otherOption(rb,path,name,device):
    model=OtherWay(rb)
    model.load(path,name)
    model.eval()
    model.cuda(device)
    return model

def load_Inapint_only(path,name,device):
    Inpaint=EdgeConn_Inpaint()
    Inpaint.load(path,name)
    Inpaint.cuda(device)
    Inpaint.eval()
    return Inpaint

def load_Edge_Inpaint(path,name1,name2,device):
    EdgeInpaint=EdgeConn()
    EdgeInpaint.load(path,name1,name2)
    EdgeInpaint.cuda(device)
    EdgeInpaint.eval()
    return EdgeInpaint

def load(config):
    if config["DH"]['scheme']!='None':
        H,R=load_Stego(config["DH"]['scheme'],
                    config["DH"]["path"], 
                    config["DH"]["H_name"], 
                    config["DH"]["R_name"],
                    device=config["Device"])
        print('load H and R of {}'.format(config["DH"]['scheme']))
    else:
        H=None
        R=None
    
    if config['PEEL']['scheme']!='None':
        Inpainting=load_AE(config["PEEL"]['rb'], config["PEEL"]["path"], config["PEEL"]["name"],device=config["Device"],inch=4)   
        print('load Inpainting from {}'.format(config["PEEL"]["name"]))
    else:
        Inpainting=None
    
    return H,R,Inpainting

