#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from config import *
from model import *
from utils import *
from train import *
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from math import log2
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.utils as vutils
import torch.nn.functional as nnf
torch.manual_seed(3)
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.autograd import Variable

from torchvision.utils import save_image
from PIL import Image
from skimage import io
from skimage.measure import compare_psnr
from skimage.transform import rescale, resize


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[3]:


def ProGAN_solver(test_img='Gabrielle Anwar',savedir='out_images_test'):
    savedir = 'out_images_test'
    test_img = 'gabrielle_2'
    
    nIter = 20001

    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    workers = 2
    ngpu = 1
    batch_size = 1  # just to make 4 dim
    iters = np.array(np.geomspace(10,10,nIter),dtype=int)
    fname = '../ProGAN_g/input_images/{}.jpg'.format(test_img)
    image = io.imread(fname)
    #x_test = resize(image, (int(config.Z_DIM*2), int(config.Z_DIM*2)),anti_aliasing=True,preserve_range=True,mode='reflect')
    image_transform = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize(size=(int(config.Z_DIM*2), int(config.Z_DIM*2))),
                                        transforms.ToTensor()])
    test_images = image_transform(image)
    test_images = test_images.unsqueeze(0)
    print(test_images.shape)
    #test_images = torch.Tensor(np.transpose(image[:batch_size,:,:,:],[0,3,1,2]))
    Z_DIM = 256 #latent dimensionality of GAN (fixed)
    IN_CHANNELS = 256
    CHANNELS_IMG = 3
    io.imsave('{}/gt.png'.format(savedir),(image).astype(np.uint8))
    
    genPATH = './ProGAN_g/generator.pth'
    #gen = Generator(config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG).to(config.DEVICE)
    gen = Generator(Z_DIM, IN_CHANNELS, img_channels=CHANNELS_IMG).to(device)
    # LOAD THE WEIGHTS AT THIS STAGE
    #print(type(gen.parameters()))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        )
        
    """if (device.type == 'cuda') and (ngpu > 1):
        gen = nn.DataParallel(gen, list(range(ngpu)))
        print('lol')
    if os.path.isfile(genPATH):
        if device.type == 'cuda':
            gen.load_state_dict(torch.load(genPATH))
        elif device.type=='cpu':
            gen.load_state_dict(torch.load(genPATH,map_location=torch.device('cpu')))
        else:
            raise Exception("Unable to load model to specified device")

        print("************ Generator weights restored! **************")
        gen.eval()"""
    gen.eval()
    #criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss()
    FIXED_NOISE = torch.randn(1, Z_DIM, 1, 1,requires_grad=True, device=device)
    #z_prior = torch.zeros(165,Z_DIM,1,1,requires_grad=True,device=device)  # config.LEARNING_RATE
    optimizerZ = optim.RMSprop([FIXED_NOISE], lr=0.02)
    #not_normal_opt = optim.Adam([FIXED_NOISE], lr=3e-4, betas=(0.0, 0.99))
    real_cpu = test_images.to(device)
    
    for iters in range(nIter):
        optimizerZ.zero_grad()
        #opt_gen.zero_grad()
        #not_normal_opt.zero_grad()
        z2 = torch.clamp(FIXED_NOISE,-1.,1.)
        #z2 = FIXED_NOISE
        fake = 0.5*gen(z2, 0.01,int(log2(config.START_TRAIN_AT_IMG_SIZE / 4)))+0.5
        cost = 0
        for i in range(3):
            y_gt = real_cpu[:,i,:,:]
            y_est = fake[:,i,:,:]
            cost += criterion(y_gt,y_est)
        
        cost.backward()
        optimizerZ.step()
        #not_normal_opt.step()
        #opt_gen.step()

        if (iters % 100 == 0):

            with torch.no_grad():
                z2 = torch.clamp(FIXED_NOISE,-1.,1.)
                #z2 = FIXED_NOISE
                fake = 0.5*gen(z2, 0.01,int(log2(config.START_TRAIN_AT_IMG_SIZE / 4))).detach().cpu() + 0.5
                G_imgs = np.transpose(fake.detach().cpu().numpy(),[0,1,2,3])
                G_imgs = torch.Tensor(G_imgs)
            
            
            #psnr = compare_psnr(x_test_,G_imgs,data_range=1.0)
            print('Iter: {:d}, Error: {:.6f}'.format(iters,cost.item()))
            #io.imsave('{}/inv_solution_iters_{}.png'.format(savedir,str(iters).zfill(4)),(G_imgs).astype(np.uint8))
            #save_image(G_imgs, f'out_images_test/inv_solution_iters_{iters}.png')
            save_image(G_imgs,'{}/inv_solution_iters_{}.png'.format(savedir, iters))


# In[4]:


if __name__ == '__main__':
    
    ProGAN_solver()


# In[ ]:


lol = int(log2(512 / 4))


# In[ ]:


print(lol)


# In[ ]:




