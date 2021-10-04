import random 
from PIL import Image
from torchvision import transforms
import torch.utils.data as data
import os
import torch
from skimage.feature import canny
import numpy as np
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def load_img(filepath):
    img = Image.open(filepath)
    return img

def input_transform(type_=0):
        if type_==0:
            tran=transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor()])
            print('Images are normalized into [0,1]')
            return tran
        # elif type_==1:
        #     tran=transforms.Compose([
        #     transforms.Resize((256,256)),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        # ])
        #     print('Secret and cover images are normalized into [-1,1]')
        #     return tran
        else:
            raise ValueError

#put all images in the same file (no subfile)
class Dataset_(data.Dataset):
    def __init__(self, image_dir='/data0/lhc/dataset/img_align_celeba',type_=0,end=2000,scheme='other',shuffle=False,max_k=80,min_k=30):
        self.max_k=80
        self.min_k=30
        super(Dataset_, self).__init__()
        self.input_transform = input_transform(type_)
        self.image_filenames = [os.path.join(image_dir, x) for x in os.listdir(image_dir) if is_image_file(x)]  
        if shuffle:
            random.shuffle(self.image_filenames)
        self.image_filenames=self.image_filenames[0:end]
        self.secret_filenames = self.image_filenames[:len(self.image_filenames)//2]
        self.cover_filenames = self.image_filenames[len(self.image_filenames)//2:]
        self.scheme=scheme
    def __getitem__(self, index):
        if self.scheme=='ISGAN':
            secret = load_img(self.secret_filenames[index])
            secret_gray = load_img(self.secret_filenames[index]).convert('L')
            cover = load_img(self.cover_filenames[index])
            secret = self.input_transform(secret)
            secret_gray=self.input_transform(secret_gray)
            cover = self.input_transform(cover)
            return secret, secret_gray, cover

        elif self.scheme=='other': #UDH, DS in test
            secret = load_img(self.secret_filenames[index])
            cover = load_img(self.cover_filenames[index])
            secret = self.input_transform(secret)
            cover = self.input_transform(cover)
            return secret, cover
        
        elif self.scheme=='AE' or self.scheme=='UDH':
            img = load_img(self.image_filenames[index])
            img=self.input_transform(img)
            return img
        elif self.scheme=='Inpainting':
            img = load_img(self.image_filenames[index])
            img=self.input_transform(img)
            gray_img=img[0,:,:]*0.299+img[1,:,:]*0.587+img[2,:,:]*0.114
            edge_canny=torch.from_numpy(canny(gray_img.detach().cpu().numpy(), sigma=1.5).astype(np.float))
            edge_canny=edge_canny.view((1,256,256))
            size=np.random.randint(self.min_k,self.max_k)
            coordinate=np.random.randint(0,128-size,2)
            x=coordinate[0]
            y=coordinate[1]
            mask=torch.zeros(1,256,256)
            mask[:,x:x+size,y:y+size]=1
            mask[:,x+128:x+size+128,y:y+size]=1
            mask[:,x:x+size,y+128:y+size+128]=1
            mask[:,x+128:x+size+128,y+128:y+size+128]=1
            img_noise_part=img.clone().detach()+torch.randn((3,256,256))*torch.randint(1,30,(1,))/100*mask
            return img.float(),edge_canny.float(),mask.float(),img_noise_part.float()
        else:
            raise ValueError
    def __len__(self):
        return len(self.secret_filenames)

# if __name__=="__main__":
#     image_dir='/data0/lhc/dataset/img_align_celeba'
#     image_filenames = [os.path.join(image_dir, x) for x in os.listdir(image_dir) if is_image_file(x)]
#     image_filenames=image_filenames[0:2000]
#     secret_filenames =image_filenames[:len(image_filenames)//2]
#     cover_filenames = image_filenames[len(image_filenames)//2:]
#     for i in range(len(secret_filenames)):
#         s_name=secret_filenames[i]
#         c_name=cover_filenames[i]
#         string='{} {} {}\n'.format(i,s_name,c_name)
#         with open('recorde.txt','a') as f:
#             f.write(string)
