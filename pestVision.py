# Copied with edits from Olivia Lamm's notebook, Kaggle
# https://www.kaggle.com/code/iamolivia/pytorch-vit-insect-classifier/data

# IP102 dataset - Kaggle

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" #use to fix libiomp5md.dll error
import util

#import numpy as np
import pandas as pd

import random
#from tqdm import tqdm
from textwrap import wrap

import torch
#import torch.nn as nn
from torch.utils.data import DataLoader#, Dataset

#import cv2 # cpu computer vision package
import matplotlib.pyplot as plt
#import seaborn as sns

#import albumentations as A
#from albumentations.pytorch.transforms import ToTensorV2

#import timm

f = open('Data\classes.txt') #local file path
label = [] # empty list creation
name = []
for line in f.readlines():
    label.append(int(line.split()[0]))
    name.append(' '.join(line.split()[1:]))
classes = pd.DataFrame([label, name]).T
classes.columns = ['label','name']
print(classes.head())

#read csv/txt into pandas dataframe
train_df = pd.read_csv(r'Data\train.txt',sep=' ',header=None, engine='python') # include r before string to avoid escape characters
train_df.columns = ['image_path','label']

test_df = pd.read_csv(r'Data\test.txt',sep=' ',header=None, engine='python')
test_df.columns = ['image_path','label']

val_df = pd.read_csv(r'Data\val.txt',sep=' ',header=None, engine='python')
val_df.columns = ['image_path','label']

print(train_df.head())

# folder paths, including r makes the 'raw'
TRAIN_DIR = r'Data\classification\train'
TEST_DIR = r'Data\classification\test'
VAL_DIR = r'Data\classification\val'
LR = 2e-5
BATCH_SIZE = 8 # number of training examples per pass
EPOCH = 2 # each epoch is a pass of all training examples

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available
device = torch.device('cpu') # i keep running out of memory (2GB gpu), running on cpu for now
# could resize images for memory constraints?

# preview random images
fig, axs = plt.subplots(3,4,figsize=(50,50)) # upping sizing a bit to better inspect
images = []
for i in classes.label:
    random_img = random.choice(train_df[train_df.label==i-1].image_path.values) # choose images randomly
    label = classes.name[i-1] # label images
    img = plt.imread(os.path.join(TRAIN_DIR,str(i-1),random_img))
    images.append(img)

[ax.imshow(image) for image,ax in zip(images,axs.ravel())]
[ax.set_title("\n".join(wrap(label,20))) for label,ax in zip(list(classes.name),axs.ravel())] # set title
[ax.set_axis_off() for ax in axs.ravel()] # set axis label
plt.show() # show plot

#########################################################################################################

# ** probably will work, havent allowed full training on model
# will take some time

# some notes
# learn about optimizers, models
# could we resize for memory efficiency?
# test differnent configs?
# better accuracy from different transformations
# the orginal code is for ViT Vision Transformer a model trained on ImageNet-21k
# https://huggingface.co/google/vit-base-patch16-224

# getting 'ibpng warning ICCP: known incorrect sRGB profile


train_dataset = util.InsectDataset(image=train_df.values, 
                              image_dir=TRAIN_DIR, 
                              transforms=util.train_transform())

train_data_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=2)

val_dataset = util.InsectDataset(image=val_df.values,
                            image_dir=VAL_DIR,
                            transforms=util.valid_transform())
                            
val_data_loader = DataLoader(val_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=True,
                             num_workers=2)

# expect to run for a long time, >10 hours on 1 epoch on CPU
util.run(device, LR, EPOCH, BATCH_SIZE, train_data_loader, val_data_loader)

model = util.InsectModel(num_classes=102) # could we filter the classes?
model.load_state_dict(torch.load("./vit_best.pth")) # load model
images, labels = next(iter(val_data_loader))
preds = model(images).softmax(1).argmax(1)

fig, axs = plt.subplots(2,4,figsize=(13,8))
[ax.imshow(image.permute((1,2,0))) for image,ax in zip(images,axs.ravel())]
[ax.set_title("\n".join(wrap(f'Acctual: {classes.name[label.item()]} Predicted: {classes.name[pred.item()]}',30)),color = 'g' if label.item()==pred.item() else 'r') for label,pred,ax in zip(labels,preds,axs.ravel())]
[ax.set_axis_off() for ax in axs.ravel()]
plt.show()

# model = util.InsectModel(num_classes=102)
# model.load_state_dict(torch.load("./vit_best.pth"))
# images, labels = next(iter(val_data_loader))
# preds = model(images).softmax(1).argmax(1)

# fig, axs = plt.subplots(2,4,figsize=(13,8))
# [ax.imshow(image.permute((1,2,0))) for image,ax in zip(images,axs.ravel())]
# [ax.set_title("\n".join(wrap(f'Acctual: {classes.name[label.item()]} Predicted: {classes.name[pred.item()]}',30)),color = 'g' if label.item()==pred.item() else 'r') for label,pred,ax in zip(labels,preds,axs.ravel())]
# [ax.set_axis_off() for ax in axs.ravel()]
# plt.show()