import argparse
import train_AE,train_DS,train_UDH,train_ISGAN,train_inpainting
import os
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scheme', default='ISGAN', choices=['DS','ISGAN','UDH','AE','Inpainting'],help='which model will be trained',)
    parser.add_argument('--data_path', default='/data0/lhc/dataset/img_align_celeba/',help='dataset path')
    parser.add_argument('--weight_path', default='checkpoint',help='path for saved models')
    parser.add_argument('--init_hide', default=None,help='pre-trained hiding models')
    parser.add_argument('--init_reveal', default=None,help='pre-trained reveal models')
    parser.add_argument('--init_dis', default=None,help='pre-trained discriminators')
    parser.add_argument('--init_ae', default='./AE_RB5.pth',help='pre-trained auto-encoder')
    parser.add_argument('--epoches', default=200, help='epoches of training')
    parser.add_argument('--enhanced', default=False, help='In deep hiding, if enhanced is True, the models are adversarially trained with pre-trained auto-encoder; In inpainting, if enhanced is True, we provide DR to the inpainting model.')
    parser.add_argument('--example_path', default='./examples', help='epoches of training')
    parser.add_argument('--bs', default=10, help='batch size')
    parser.add_argument('--rb', default=8, help='number of residual blocks in auto-encoder and inpainting model')
    parser.add_argument('--num_c', default=1, help='the number of covers for each secret image')
    parser.add_argument('--GPUs', default='1', help='which GPU will be used, e.g. \'0,1\'')
    opt = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.GPUs
    if opt.scheme=='UDH':
        print('train UDH models')
        train_UDH.main(opt.data_path,opt.weight_path,opt.init_hide,opt.init_reveal,opt.init_ae,opt.epoches,opt.enhanced,os.path.join(opt.example_path,opt.scheme),opt.bs,opt.num_c)
    elif opt.scheme=='DS':
        print('train DS models')
        train_DS.main(opt.data_path,opt.weight_path,opt.init_hide,opt.init_reveal,opt.init_ae,opt.epoches,opt.enhanced,os.path.join(opt.example_path,opt.scheme),opt.bs)
    elif opt.scheme=='ISGAN':
        print('train ISGAN models')
        train_ISGAN.main(opt.data_path,opt.weight_path,opt.init_hide,opt.init_reveal,opt.init_dis,opt.init_ae,opt.epoches,opt.enhanced,os.path.join(opt.example_path,opt.scheme),opt.bs)
    elif opt.scheme=='AE':
        print('train auto-encoder')
        train_AE.main(opt.data_path,opt.weight_path,opt.epoches,os.path.join(opt.example_path,opt.scheme),opt.bs,opt.rb)
    elif opt.scheme=='Inpainting':
        print('train Inpainting model')
        train_inpainting.main(opt.data_path,opt.weight_path,opt.epoches,os.path.join(opt.example_path,opt.scheme),opt.enhanced,opt.bs,opt.rb)





