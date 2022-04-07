import os # library for os dependent functions
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" #use to fix libiomp5md.dll error

import torch
import util
import matplotlib.pyplot as plt # library for creating visuals
from textwrap import wrap # wraps text to n characters
from torch.utils.data import Dataset, DataLoader
import pandas as pd

#f = open('Data\classes.txt') #local file path
f = open(r'Data\newClasses.txt') #local file path
label = [] # empty list creation
name = []
for line in f.readlines():
    label.append(int(line.split()[0]))
    name.append(' '.join(line.split()[1:]))
classes = pd.DataFrame([label, name]).T # '.T' transpose index and columns
classes.columns = ['label','name'] # sets column names

VAL_DIR = r'Data\classification\val'
LR = 2e-5 # learning rate, could set up a LR scheduler?
BATCH_SIZE = 8 # number of training examples per pass
EPOCH = 2 # each epoch is a pass of all training examples

val_df = pd.read_csv(r'Data\newval.txt',sep=' ',header=None, engine='python')
val_df.columns = ['image_path','label']

val_dataset = util.InsectDataset(image=val_df.values,
                            image_dir=VAL_DIR,
                            transforms=util.valid_transform())
                            
val_data_loader = DataLoader(val_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=True,
                             num_workers=2)

model = util.InsectModel(num_classes=len(classes)) # could we filter the classes?
model.load_state_dict(torch.load(r"Model\vit.pth")) # load model
images, labels = next(iter(val_data_loader))
preds = model(images).softmax(1).argmax(1)

fig, axs = plt.subplots(2,4,figsize=(13,8))
[ax.imshow(image.permute((1,2,0))) for image,ax in zip(images,axs.ravel())]
[ax.set_title("\n".join(wrap('Actual: {classes.name[label.item()]} Predicted: {classes.name[pred.item()]}',30)),color = 'g' if label.item()==pred.item() else 'r') for label,pred,ax in zip(labels,preds,axs.ravel())]
[ax.set_axis_off() for ax in axs.ravel()]
plt.show()

model = util.InsectModel(num_classes=len(classes))
model.load_state_dict(torch.load(r"Model\\vit.pth"))
images, labels = next(iter(val_data_loader))
preds = model(images).softmax(1).argmax(1)

fig, axs = plt.subplots(2,4,figsize=(13,8))
[ax.imshow(image.permute((1,2,0))) for image,ax in zip(images,axs.ravel())]
[ax.set_title("\n".join(wrap(f'Actual: {classes.name[label.item()]} Predicted: {classes.name[pred.item()]}',30)),color = 'g' if label.item()==pred.item() else 'r') for label,pred,ax in zip(labels,preds,axs.ravel())]
[ax.set_axis_off() for ax in axs.ravel()]
plt.show()