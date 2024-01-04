#!/usr/bin/env python
# coding: utf-8

# ## This code is written in Pytorch for the DISK project (NO Labels). 
# @author: Sayantan
# date: 2 Jan 2024
# 
# ### The main idea is to use the Convolution Network and check if that performs better than the traditional 
# ###  MLP model that was used earlier
# 
# This notebook is adopted from the GAN_DISK_CODE-V3
# The idea is to rotate disk images from any arbitary angles to face on images
# We shall use the DPPNET-Rt synthetic dataset for this purpose
# 
# 
# Improvement from the V1 script is that: We now consider the data from the 700 different
# simulations to check if the output is dependent on the input data, i.e., the characteristic of the image
# and not just the rotation angle
# 
# 
# 

# In[1]:


import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import sys
import platform
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import os
import csv
import re ## For data manipulation
from sklearn.model_selection import train_test_split

from PIL import Image, ImageOps

has_gpu = torch.cuda.is_available()
has_mps = torch.backends.mps.is_built()
device = "mps" if torch.backends.mps.is_built()     else "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"Python Platform: {platform.platform()}")
print(f"PyTorch Version: {torch.__version__}")

# print(f"Scikit-Learn {sk.__version__}")
print("GPU is", "available" if has_gpu else "NOT AVAILABLE")
print("MPS (Apple Metal) is", "AVAILABLE" if has_mps else "NOT AVAILABLE")
print(f"Target device is {device}")


# In[2]:


parser = argparse.ArgumentParser()
parser.add_argument('-f')
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling")
opt = parser.parse_args()
print(opt)


# In[3]:


def create_complete_data_csv(list_sorted_RT_path ,path):
    '''
    Input : list of address to the RT image folder, example "RT_A_1"
            Path to the working directory where the cluster_run_1.csv is and the data_folder is created
    
    Output : Return a conacted csv with sim parameters and path to the RT images 
             (optional)Returns csv for each simulations along with the path to the augmented images 
    
    '''
    
    dataset_complete = []
    print("[INFO]: Importing path for all the RT images")
    for index in range(len(list_sorted_RT_path)):

        path_image = list_sorted_RT_path[index] ## path to each RT folder
        ## for paths from the RT sim fodlers directly
        list_image_path = glob.glob(path_image +  "/image_"+"*.png") ## list of the path to each image in the RT folder
        
        list_sorted_image_path  = sorted(list_image_path, key=lambda f: [int(n) for n in re.findall(r"\d+", f)][-1])      
#         print(list_sorted_RT_path)                             
        df_images_folder =pd.DataFrame(list_sorted_image_path,columns=["image_path"]) ## making a dataframe with the images path
        df_images_folder ['target'] = df_images_folder['image_path'].iloc[0]
                                                      
        # Appending the data from the pandas dataframe for each orbits
        dataset_complete.append(df_images_folder)

    dataset_complete = pd.concat(dataset_complete, ignore_index=True, axis=0)  
    dataset_complete.to_csv(path+'data_folder/dataset_complete.csv') 
    return  dataset_complete


# In[4]:


############# Address to the data folder ###################
current_directory = os.getcwd()
os.makedirs("data_folder", exist_ok=True)
path = current_directory + '/' # For local computer 
# ## updating the image paths once the transfer is done
try:     
    list_RT_path = glob.glob(path+ '../DPNNet-RT/DPNNET-RT-3Nov22/image_directory_complete/'+ 'RT_A*') ## make a list of all the RT folder where each folder is for each sim
    list_sorted_RT_path  = sorted(list_RT_path, key=lambda f: [int(n) for n in re.findall(r"\d+", f)][-1]) ## sorting the images
#     print(list_sorted_RT_path)
  
    data_complete = create_complete_data_csv(list_sorted_RT_path,path)
except ValueError:
    print("Looking for images in the folder above-- Please give the correct path to the images if images are not loaded")
    list_RT_path = glob.glob(path+ '../image_directory_complete/'+ 'RT_A*') ## make a list of all the RT folder where each folder is for each sim
    # ## For google colab this needs to be updated
    list_sorted_RT_path  = sorted(list_RT_path, key=lambda f: [int(n) for n in re.findall(r"\d+", f)][-1]) ## sorting the images
    data_complete = dp.create_complete_data_csv(list_sorted_RT_path,path)
    
    
complete_dataset_mod = data_complete
## Split the dataset and only consider 30 percent of the data:
split = train_test_split(complete_dataset_mod, test_size=0.70, random_state=42)
dataset, dataset_unused = split

# complete_dataset_mod['target'] = complete_dataset_mod['image_path'].iloc[9]


# In[5]:


dataset


# In[6]:


## Splitting the data for training and testing 

split_1 = train_test_split(dataset, test_size=0.15, random_state=42)
(train, test) = split_1
train.to_csv(path+'data_folder/train.csv') 


# In[7]:


## Defining a custom class that returns image tensor and the corresponding label for a given index

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,dataframe,transform = None):
        self.df = dataframe
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        filename = self.df["image_path"].iloc[index]
        target_filename = self.df["target"].iloc[index]
        image = Image.open(filename)
        target_image = Image.open(target_filename)
        
        if opt.channels ==3:
            image = image.convert('RGB')#('L')# ## Converting to gray-scale with one channel        
            target_image = target_image.convert('RGB')#('L')#
        elif opt.channels ==1:
            image = image.convert('L')# ## Converting to gray-scale with one channel        
            target_image = target_image.convert('L')#
#         print("THE SHAPE OF THE IMAGE",np.shape(image))
        left = 105
        top = 55
        right = 480
        bottom = 430 
        # Cropped image of above dimension" # (It will not change original image)
        image = image.crop((left, top, right, bottom))
        target_image = target_image.crop((left, top, right, bottom))
        if self.transform is not None:
            image = self.transform(image)
            target_image = self.transform(target_image)
#         print(np.shape(image))
        return image,target_image


# In[8]:


## To verify if the images are read correctly.
image_output = CustomDataset(complete_dataset_mod,transform= None)
# image_output = CustomDataset(train,transform = None)
image ,target_image= image_output.__getitem__(243)
# target_image
image


# In[9]:


transform_custom = transforms.Compose([transforms.Resize(opt.img_size),
   transforms.ToTensor()])
image_output = CustomDataset(train,transform= transform_custom)
# image_output = CustomDataset(train,transform = None)
image ,target_image= image_output.__getitem__(10)
print(np.shape(image),np.shape(target_image))
# gray_image = ImageOps.grayscale(image)
# print(type(gray_image))
image.data, target_image.data

save_image(target_image.data, "test.png", normalize=True)


# In[10]:


# IMG_SIZE = opt.img_size
BATCH_SIZE = opt.batch_size
# Parameters
params = {'batch_size':opt.batch_size ,
          'shuffle': True}


# In[11]:


training_set = CustomDataset(train,transform=transform_custom)
training_generator = torch.utils.data.DataLoader(training_set, **params)


# In[12]:


# opt.channels, opt.img_size, opt.img_size = 1 ,32,32
img_shape = (opt.channels, opt.img_size, opt.img_size)
print(img_shape)


# In[13]:


# np.prod(img_shape)


# In[14]:


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


# In[15]:


# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()


# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 100

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_size// 2 ** 4, opt.img_size // 2 ** 4)


# Initialize generator and discriminator
generator = GeneratorUNet()
discriminator = Discriminator()

generator.to(device)
discriminator.to(device)
criterion_GAN.to(device)
criterion_pixelwise.to(device)


# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


# In[16]:


test.to_csv(path+'data_folder/test.csv') 


# In[17]:


def sample_image(batch_size , batches_done=None):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample images
    params_test = {'batch_size':batch_size  ,
          'shuffle': True}
    test_set = CustomDataset(test[:batch_size],transform=transform_custom)
    test_generator = torch.utils.data.DataLoader(test_set, **params_test )
    
    
    
    for i, (imgs,targets) in enumerate(test_generator):
        sample_size = imgs.shape[0]
        print("shape of image",np.shape(imgs))
        
        input_imgs = Variable(imgs.type(torch.FloatTensor).to(device))   
        print("shape of input image",np.shape(input_imgs))
        gen_imgs = generator(input_imgs)
        
        
        
        
        print("shape of generated image",np.shape(gen_imgs))
        print(type(gen_imgs),type(imgs))
        cat_images = torch.cat((targets.to(device),gen_imgs.to(device),imgs.type(torch.FloatTensor).to(device)),axis=0)
        
#         print(np.shape(cat_images))
#         save_image(gen_imgs.data, "images/test.png", normalize=True)
#         save_image(imgs.data,"images/input.png" ,nrow=10, normalize=True)
    
        save_image(gen_imgs.data, "images/%d.png" % batches_done,nrow=batch_size ,normalize=True)
        save_image(cat_images.data, "images/input_target%d.png" % batches_done,nrow=batch_size ,normalize=True)
    

        


# In[18]:


os.makedirs("images", exist_ok=True)
sample_image(batch_size=10,batches_done=100)


# In[ ]:


for epoch in range(opt.n_epochs):
    # Training
    for i, (imgs,target_image) in enumerate(training_generator):
        print(type(imgs))
        batch_size = imgs.shape[0]
  
    
        # INPUT INCLIDED IMAGE
        input_imgs = Variable(imgs.type(torch.FloatTensor).to(device))
        
        # TARGET IMAGE
        face_on_image = Variable(target_image.type(torch.FloatTensor).to(device))
        
        # Adversarial ground truths
        valid = Variable(torch.FloatTensor(input_imgs.size(0), *patch).fill_(1.0).to(device), requires_grad=False)
        fake  = Variable(torch.FloatTensor(face_on_image.size(0), *patch).fill_(0.0).to(device), requires_grad=False)

        
        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        
        
        ######  # GAN loss ##############
        
        # Generate a batch of images
        gen_imgs = generator(input_imgs)
        pred_fake = discriminator(gen_imgs, input_imgs)
        
        loss_GAN = criterion_GAN(pred_fake, valid)
        
#         print(np.shape(gen_imgs),np.shape(face_on_image))
        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(gen_imgs, face_on_image)
        
        

        # Total loss
        loss_G = loss_GAN + lambda_pixel * loss_pixel

        loss_G.backward()

        optimizer_G.step()
        
        
        # Real loss
        pred_real = discriminator(face_on_image, input_imgs)
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss
        pred_fake = discriminator(gen_imgs.detach(), input_imgs)
        loss_fake = criterion_GAN(pred_fake, fake)

        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------
       

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(training_generator), loss_D.item(), loss_G.item())
        )


        batches_done = epoch * len(training_generator) + i
#         print(epoch,batches_done,len(training_generator))
#         print(np.shape(gen_imgs.data))
        if batches_done % opt.sample_interval == 0:
#             save_image(gen_imgs.data[:2], "images/%d.png" % batches_done, nrow=3, normalize=True)
            sample_image(batch_size=8, batches_done=batches_done)


# In[ ]:


# image_output = CustomDataset(test,transform= transform_custom)
# # image_output = CustomDataset(train,transform = None)
# test_image10,_= image_output.__getitem__(16)
# test_image11,_= image_output.__getitem__(3)

# print(np.shape(test_image10),np.shape(test_image11))
# test_cat=torch.cat((test_image10,test_image10),axis=0)
# print(np.shape(test_cat))
# # test


# In[ ]:


# input_imgs = Variable(test_cat.type(torch.FloatTensor).to(device)).reshape(2,int(np.prod(img_shape)))
# print(np.shape(input_imgs))
# gen_imgs = generator(input_imgs)


# In[ ]:


# gen_imgs = generator(input_imgs)
# print(type(gen_imgs),type(test_cat))
# # print(np.shape(gen_imgs),np.shape(test_cat),np.shape(test_cat.reshape(2,1,256,256)))
# save_image(test_image10.data, "images/in10.png", normalize=True)
# save_image(test_image11.data, "images/in11.png", normalize=True)

# save_image(gen_imgs.data, "images/test1.png", normalize=True)


# In[ ]:




