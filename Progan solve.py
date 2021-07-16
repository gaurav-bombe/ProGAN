#!/usr/bin/env python
# coding: utf-8

# In[29]:


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
torch.manual_seed(0)
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


# In[30]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[31]:


def ProGAN_solver(test_img='Gabrielle Anwar',savedir='out_images_test'):
    savedir = 'out_images_test'
    test_img = 'Gabrielle Anwar'
    
    nIter = 101

    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    workers = 2
    ngpu = 1
    batch_size = 1  # just to make 4 dim
    iters = np.array(np.geomspace(10,10,nIter),dtype=int)
    fname = '../ProGAN/input_images/{}.jpg'.format(test_img)
    image = io.imread(fname)
    image_transform = transforms.Compose([transforms.ToPILImage(),
                                     transforms.ToTensor()])
    test_images = image_transform(image)
    test_images = test_images.unsqueeze(0)
    print(test_images.shape)
    #test_images = torch.Tensor(np.transpose(image[:batch_size,:,:,:],[0,3,1,2]))
    Z_DIM = 256 #latent dimensionality of GAN (fixed)
    IN_CHANNELS = 256
    CHANNELS_IMG = 3
    io.imsave('{}/gt.png'.format(savedir),(image).astype(np.uint8))
    
    genPATH = './ProGAN/generator.pth'
    #gen = Generator(config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG).to(config.DEVICE)
    gen = Generator(Z_DIM, IN_CHANNELS, img_channels=CHANNELS_IMG).to(device)
    # LOAD THE WEIGHTS AT THIS STAGE
    if (device.type == 'cuda') and (ngpu > 1):
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
        gen.eval()
    
    criterion = nn.MSELoss()
    FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)
    #z_prior = torch.zeros(165,Z_DIM,1,1,requires_grad=True,device=device)
    optimizerZ = optim.RMSprop([FIXED_NOISE], lr=5e-3)

    real_cpu = test_images.to(device)
    
    for iters in range(nIter):
        optimizerZ.zero_grad()

        z2 = torch.clamp(FIXED_NOISE,-1.,1.)
        fake = 0.5*gen(z2, 1e-5,int(log2(config.START_TRAIN_AT_IMG_SIZE / 4)))+0.5
        cost = 0
        for i in range(3):
            y_gt = real_cpu[:,i,:,:]
            y_est = fake[:,i,:,:]
            cost += criterion(y_gt,y_est)
        
        cost.backward()
        optimizerZ.step()
        if (iters % 50 == 0):

            with torch.no_grad():
                z2 = torch.clamp(FIXED_NOISE,-1.,1.)
                fake = 0.5*gen(z2, 1e-5,int(log2(config.START_TRAIN_AT_IMG_SIZE / 4))).detach().cpu() + 0.5
                G_imgs = np.transpose(fake.detach().cpu().numpy(),[0,2,3,1])
            
            
            #psnr = compare_psnr(x_test_,G_imgs,data_range=1.0)
            print('Iter: {:d}, Error: {:.3f}'.format(iters,cost.item()))
            io.imsave('{}/inv_solution_iters_{}.png'.format(savedir,str(iters).zfill(4)),(G_imgs).astype(np.uint8))


# In[32]:


if __name__ == '__main__':
    
    ProGAN_solver()


# In[ ]:




