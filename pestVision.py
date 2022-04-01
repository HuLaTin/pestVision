#copied from Olivia Lamm's code from Kaggle
#IP102 dataset - Kaggle

import numpy as np
import pandas as pd
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" #use to fix libiomp5md.dll error

import random
from tqdm import tqdm
from textwrap import wrap

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import cv2 # cpu computer vision package
import matplotlib.pyplot as plt
import seaborn as sns

import albumentations as A #not familiar with this error
from albumentations.pytorch.transforms import ToTensorV2

import timm

f = open('Data\classes.txt') #local file path
label = [] # empty lists
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

#folder paths, including r makes the string literal
TRAIN_DIR = r'Data\classification\train'
TEST_DIR = r'Data\classification\test'
VAL_DIR = r'Data\classification\val'
LR = 2e-5
BATCH_SIZE = 8
EPOCH = 2

device = torch.device('cuda') # CUDA for use with GPU, include for cpu?

# preview random images
fig, axs = plt.subplots(10,11,figsize=(30,30))
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