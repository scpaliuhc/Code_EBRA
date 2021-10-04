from collections import OrderedDict
import torch

def load_model_for_par(path):
    #remove 'module.' from the keys
    state_dict=torch.load(path)
    for key in state_dict.keys():
        None
    if key[:7]=='module.':  
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] 
            new_state_dict[name] = v
        return new_state_dict
    else:
        return state_dict

def removal_inpaint_only(stego,inpaint_model,edge_canny,s=20,t=20,device='cuda:0'):
    print('PEEL')
    stego_copy=stego.clone().detach()
    for h in range(0,128,s):
        for w in range(0,128,s):
            mask=torch.zeros((1,1,256,256)).cuda(device)
            mask[:,:,h:h+t,w:w+t]=1
            mask[:,:,h+128:h+128+t,w:w+t]=1
            mask[:,:,h:h+t,w+128:w+128+t]=1
            mask[:,:,h+128:h+128+t,w+128:w+128+t]=1
            stego_mask=stego*(1-mask)+mask
            # stego_mask=stego_copy*(1-mask)+mask ###
            inputs=torch.cat([stego_mask,edge_canny],dim=1)
            output=inpaint_model(inputs)
            stego_copy[:,:,h:h+t,w:w+t]=output[:,:,h:h+t,w:w+t]
            stego_copy[:,:,h+128:h+128+t,w:w+t]=output[:,:,h+128:h+128+t,w:w+t]
            stego_copy[:,:,h:h+t,w+128:w+128+t]=output[:,:,h:h+t,w+128:w+128+t]
            stego_copy[:,:,h+128:h+128+t,w+128:w+128+t]=output[:,:,h+128:h+128+t,w+128:w+128+t]
    return stego_copy


def removal_edge_inpaint(stego,stego_gray,edge_inpaint_model,edge_canny,s=20,t=20,device='cuda:0'):
    print('edge_inpaint')
    stego_copy=stego.clone().detach()
    for h in range(0,128,s):
        for w in range(0,128,s):
            mask=torch.zeros((1,1,256,256)).cuda(device)
            mask[:,:,h:h+t,w:w+t]=1
            mask[:,:,h+128:h+128+t,w:w+t]=1
            mask[:,:,h:h+t,w+128:w+128+t]=1
            mask[:,:,h+128:h+128+t,w+128:w+128+t]=1
            output=edge_inpaint_model(stego_gray,stego,mask,edge_canny)
            stego_copy[:,:,h:h+t,w:w+t]=output[:,:,h:h+t,w:w+t]
            stego_copy[:,:,h+128:h+128+t,w:w+t]=output[:,:,h+128:h+128+t,w:w+t]
            stego_copy[:,:,h:h+t,w+128:w+128+t]=output[:,:,h:h+t,w+128:w+128+t]
            stego_copy[:,:,h+128:h+128+t,w+128:w+128+t]=output[:,:,h+128:h+128+t,w+128:w+128+t]
    return stego_copy

def removal_blur(stego,blur_model,edge_canny,s=40,t=40,device='cuda:0',k_size=3,sigma=5):
    print('blurpart')
    stego_copy=stego.clone().detach()
    import cv2
    stego_copy_blur=cv2.GaussianBlur(stego.clone().detach().cpu().numpy()[0].transpose((1,2,0)),(k_size,k_size),sigma)#5
    stego_copy_blur=torch.from_numpy(stego_copy_blur.transpose((2,0,1))).type_as(stego_copy).unsqueeze(0).cuda(device)

    # img=stego.cpu().clone().detach().numpy()[0].transpose((1,2,0))
    # imgblur=cv2.GaussianBlur(img,(5,5),1)
    # stego_blur=torch.from_numpy(imgblur.transpose(2,0,1)).unsqueeze(0).cuda(device)
    #不累计
    for h in range(0,128,s):
        for w in range(0,128,s):
            stego_mask=stego.clone().detach()
            # stego_mask=stego_blur.clone().detach()
            stego_mask[:,:,h:h+t,w:w+t]=stego_copy_blur[:,:,h:h+t,w:w+t]
            stego_mask[:,:,h+128:h+128+t,w:w+t]=stego_copy_blur[:,:,h+128:h+128+t,w:w+t]
            stego_mask[:,:,h:h+t,w+128:w+128+t]=stego_copy_blur[:,:,h:h+t,w+128:w+128+t]
            stego_mask[:,:,h+128:h+128+t,w+128:w+128+t]=stego_copy_blur[:,:,h+128:h+128+t,w+128:w+128+t]
            inputs=torch.cat([stego_mask,edge_canny],dim=1)
            output=blur_model(inputs)   
            stego_copy[:,:,h:h+t,w:w+t]=output[:,:,h:h+t,w:w+t]
            stego_copy[:,:,h+128:h+128+t,w:w+t]=output[:,:,h+128:h+128+t,w:w+t]
            stego_copy[:,:,h:h+t,w+128:w+128+t]=output[:,:,h:h+t,w+128:w+128+t]
            stego_copy[:,:,h+128:h+128+t,w+128:w+128+t]=output[:,:,h+128:h+128+t,w+128:w+128+t]
    #累计
    # stego_copy=stego.clone().detach()
    # for h in range(0,128,s):
    #     for w in range(0,128,s):
            
    #         # stego_mask=stego_blur.clone().detach()
    #         stego_copy[:,:,h:h+t,w:w+t]=stego_copy_blur[:,:,h:h+t,w:w+t]
    #         stego_copy[:,:,h+128:h+128+t,w:w+t]=stego_copy_blur[:,:,h+128:h+128+t,w:w+t]
    #         stego_copy[:,:,h:h+t,w+128:w+128+t]=stego_copy_blur[:,:,h:h+t,w+128:w+128+t]
    #         stego_copy[:,:,h+128:h+128+t,w+128:w+128+t]=stego_copy_blur[:,:,h+128:h+128+t,w+128:w+128+t]
    #         inputs=torch.cat([stego_copy,edge_canny],dim=1)
    #         output=blur_model(inputs)   
    #         stego_copy[:,:,h:h+t,w:w+t]=output[:,:,h:h+t,w:w+t]
    #         stego_copy[:,:,h+128:h+128+t,w:w+t]=output[:,:,h+128:h+128+t,w:w+t]
    #         stego_copy[:,:,h:h+t,w+128:w+128+t]=output[:,:,h:h+t,w+128:w+128+t]
    #         stego_copy[:,:,h+128:h+128+t,w+128:w+128+t]=output[:,:,h+128:h+128+t,w+128:w+128+t]
    
    # stego_copy_blur=stego_copy.clone().detach()
    # for h in range(0,128,s):
    #     for w in range(0,128,s):
    #         stego_mask=stego.clone().detach()
    #         stego_mask[:,:,h:h+t,w:w+t]=stego_copy_blur[:,:,h:h+t,w:w+t]
    #         stego_mask[:,:,h+128:h+128+t,w:w+t]=stego_copy_blur[:,:,h+128:h+128+t,w:w+t]
    #         stego_mask[:,:,h:h+t,w+128:w+128+t]=stego_copy_blur[:,:,h:h+t,w+128:w+128+t]
    #         stego_mask[:,:,h+128:h+128+t,w+128:w+128+t]=stego_copy_blur[:,:,h+128:h+128+t,w+128:w+128+t]
    #         inputs=torch.cat([stego_mask,edge_canny],dim=1)
    #         output=blur_model(inputs)   
    #         stego_copy[:,:,h:h+t,w:w+t]=output[:,:,h:h+t,w:w+t]
    #         stego_copy[:,:,h+128:h+128+t,w:w+t]=output[:,:,h+128:h+128+t,w:w+t]
    #         stego_copy[:,:,h:h+t,w+128:w+128+t]=output[:,:,h:h+t,w+128:w+128+t]
    #         stego_copy[:,:,h+128:h+128+t,w+128:w+128+t]=output[:,:,h+128:h+128+t,w+128:w+128+t]
    return stego_copy


def removal_DeNoise_part(stego,DeNoise_model,edge_canny,s=40,t=40,scale=0.25,device='cuda:0'):
    print('PEELO')
    stego_copy=stego.clone().detach()
    stego_Noises=stego+torch.randn((stego.shape[0],3,256,256)).cuda(device)*scale
    for h in range(0,128,s):
        for w in range(0,128,s):
            stego_mask=stego.clone().detach()        
            stego_mask[:,:,h:h+t,w:w+t]=stego_Noises[:,:,h:h+t,w:w+t]
            stego_mask[:,:,h+128:h+128+t,w:w+t]=stego_Noises[:,:,h+128:h+128+t,w:w+t]
            stego_mask[:,:,h:h+t,w+128:w+128+t]=stego_Noises[:,:,h:h+t,w+128:w+128+t]
            stego_mask[:,:,h+128:h+128+t,w+128:w+128+t]=stego_Noises[:,:,h+128:h+128+t,w+128:w+128+t]
            inputs=torch.cat([stego_mask,edge_canny],dim=1)
            output=DeNoise_model(inputs)   
            stego_copy[:,:,h:h+t,w:w+t]=output[:,:,h:h+t,w:w+t]
            stego_copy[:,:,h+128:h+128+t,w:w+t]=output[:,:,h+128:h+128+t,w:w+t]
            stego_copy[:,:,h:h+t,w+128:w+128+t]=output[:,:,h:h+t,w+128:w+128+t]
            stego_copy[:,:,h+128:h+128+t,w+128:w+128+t]=output[:,:,h+128:h+128+t,w+128:w+128+t]
    return stego_copy

def removal_Deblur(stego,blur_model,edge_canny,device):
    print('Deblur')
    # stego_copy=stego.clone().detach()
    import cv2
    #7:07094932
    #(7,7),5:07101834
    #(7,7),7:07102820
    #(9,9),7:07105110
    stego_copy_blur=cv2.GaussianBlur(stego.clone().detach().cpu().numpy()[0].transpose((1,2,0)),(7,7),7)
    stego_copy_blur=torch.from_numpy(stego_copy_blur.transpose((2,0,1))).type_as(stego).unsqueeze(0).cuda(device)
    inputs=torch.cat([stego_copy_blur,edge_canny],dim=1)
    outputs=blur_model(inputs)
    return outputs

def removal_AE(stego,AE_model):
    print('AE')
    return AE_model(stego)


def removal_DeAE(stego,AE_model,device):
    print('DeAE')
    stego_new=stego+torch.randn((stego.shape[0],3,256,256)).cuda(device)*0.25
    return AE_model(stego_new)

def Resize(stego):
    print('resize')
    import cv2
    img=stego.cpu().clone().detach().numpy()[0].transpose((1,2,0))
    img128=cv2.resize(img,(128,128))
    img256=cv2.resize(img128,(256,256))
    return torch.from_numpy(img256.transpose(2,0,1)).unsqueeze(0)

def Lattice(stego,scale,mask,RC,C,device):
    print('lattice')
    if RC=='r':
        sign1=((torch.randint(0,2,(256,256))-0.5)*2).to(device)*mask
        sign2=((torch.randint(0,2,(256,256))-0.5)*2).to(device)*mask
        stego_copy=stego.clone().detach()
        stego_copy[:,0,:,:]=stego_copy[:,0,:,:]*(1-mask)+sign2*scale
        stego_copy[:,1,:,:]=stego_copy[:,1,:,:]*(1-mask)+sign1*scale
        stego_copy[:,2,:,:]=stego_copy[:,2,:,:]*(1-mask)-sign1*scale
    if RC=='c':
        stego_copy=stego.clone().detach()
        stego_copy[:,0,:,:]=stego_copy[:,0,:,:]*(1-mask)+mask*C[0]
        stego_copy[:,1,:,:]=stego_copy[:,1,:,:]*(1-mask)+mask*C[1]
        stego_copy[:,2,:,:]=stego_copy[:,2,:,:]*(1-mask)+mask*C[2]
    # stego_new=stego+ torch.randn((stego.shape[0],channel,256,256)).cuda(device)*scale*mask
    return stego_copy

def GaussianBlur(stego,size,sigma):
    print('GaussianBlur')
    import cv2
    img=stego.cpu().clone().detach().numpy()[0].transpose((1,2,0))
    imgblur=cv2.GaussianBlur(img,(size,size),sigma)
    return torch.from_numpy(imgblur.transpose(2,0,1)).unsqueeze(0)

def MedianBlur(stego,size):
    print('MedianBlur')
    import cv2
    img=stego.cpu().clone().detach().numpy()[0].transpose((1,2,0))
    imgblur=cv2.medianBlur(img,size)
    return torch.from_numpy(imgblur.transpose(2,0,1)).unsqueeze(0)

def GaussianNoise(stego,scale,channel):
    print('GaussianNoise')
    stego_noised=stego.to('cpu')+torch.randn(1,channel,256,256)*scale
    return stego_noised

def FGSM(stego,secret,R, MSE_loss,ep):
    print('FGSM')
    stego=stego.clone().detach()
    stego.requires_grad =True
    reveal_stego=R(stego)
    loss=MSE_loss(reveal_stego,secret)
    R.zero_grad()
    loss.backward()
    data_grad_sign = stego.grad.data.sign()
    perturb_stego=stego+ep*data_grad_sign
    perturb_stego=torch.clamp(perturb_stego, 0, 1)
    return perturb_stego


