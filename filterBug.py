import os
import pandas as pd
import shutil
import numpy as np

# THIS WILL DELETE FILES
# PLEASE BACKUP YOUR DATASET

# I still have to manually rename folders 0-28

f = open('Data\classes.txt') #local file path
label = [] # empty list creation
name = []
for line in f.readlines():
    label.append(int(line.split()[0]))
    name.append(' '.join(line.split()[1:]))
#f.close() # better to close this?
classes = pd.DataFrame([label, name]).T # '.T' transpose index and columns
classes.columns = ['label','name'] # sets column names
print(classes.head())

# these are the classes we are discarding from our dataset
# filtering based on numbering in classes.txt
# this is a really poor dataset, classes of caterpillar contain images of moth, maggots contain adult flies
# manually selected classes, would be important to revisit
discard = [2,3,6,8,9,10,11,12,13,14,15,16,17,22,25,26,28,29,30, \
        31,32,33,34,35,36,37,38,41,43,44,47,48,50,51,52,53,54,55,56,58, \
        60,61,62,63,64,65,69,71,72,73,75,76,77,78,79,80,81,82,83, \
        84,85,86,89,90,91,92,93,94,95,97,99,100,101]

classes = classes.loc[~classes['label'].isin(discard)] # select rows where 'label' is not in discard list
classes['label'] = np.arange(1, len(classes) + 1)

classes.reset_index(inplace=True, drop=True) # reset index

np.savetxt(r'Data\newClasses.txt', classes.to_numpy(), fmt = "%s") # save to txt file, space as delimiter

dataDir = r'Data\classification' # setting folder path, string
folders = ['test','train','val'] # list of folders to filter

# Data\classification\test\0 # example
# oops no need to delete images
# might need to actually

# for i in folders:
#     folderPath = dataDir + '\\' + i
#     for j in discard:
#         labelPath = folderPath + '\\' + str(j - 1) # j - 1 to select the correct folder
#         shutil.rmtree(labelPath) # removes folder!

# I am manually renaming folders
#for root, dirs, files in os.walk(r"Data\classification"):
#    print(root)

discard = [i-1 for i in discard]
txtPath = r'Data'
for i in folders:
    fileToRewrite = txtPath + '\\' + str(i) + '.txt'
    f = open(fileToRewrite) # open file
    label = [] # empty list creation
    name = []
    for line in f.readlines(): # read through file
        label.append(line.split()[0])
        name.append(int(' '.join(line.split()[1:])))
    TXT = pd.DataFrame([label, name]).T
    f.close()

    print(i) # print train/test/val set
    print('images before filter : ', len(TXT))
    len(TXT[1].unique())
    TXT = TXT.loc[~TXT[1].isin(discard)]
    len(TXT[1].unique())
    print('images after filter : ', len(TXT))

    indClass = list(TXT[1].unique())
    #nextClass = iter(indClass)
    for v,z in enumerate(range(len(classes))):
        #print(indClass[v])
        #print(z)
        TXT.loc[TXT[1] == indClass[v],1] = int(z)



    TXT.reset_index(inplace=True, drop=True) # reset index

    txtName = txtPath + '\\new' + str(i) + '.txt' # file path for 'new' txt file
    np.savetxt(txtName, TXT.to_numpy(), fmt = "%s") # save it!

# entire dataset is now 32143 total images, (from ~75k)
# we're not actually deleting anything, just modifying the classes which we train/val/test on (I think)