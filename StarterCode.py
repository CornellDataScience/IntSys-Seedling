# -*- coding: utf-8 -*-
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure


# CREATING A SIMPLE CNN USING VGG16
import keras as k
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten, Input, Softmax
from keras.models import Model

#input layer description + creation
inLayer = Input(shape=(224, 224, 3), name='in_layer')
vgg = VGG16(weights = 'imagenet', include_top = False)
for l in vgg.layers :
    l.trainable = False
cnn = (vgg)(inLayer)
flat = Flatten()(cnn)
fc = Dense(12)(flat)
output = Softmax(name = 'soft')(fc)

#compile model
model = Model(inputs=inLayer, outputs = output)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())

# GENERATING TRAINING DATA 
from PIL import Image
import cv2
from pathlib import Path

#let's take 200 images from each category
trainImg = []
trainTarget = []
catNames = ["Black-grass", "Charlock", "Cleavers", "Common Chickweed", "Common wheat", "Fat Hen", "Loose Silky-bent", "Maize", "Scentless Mayweed", "Shepherd'd Purse", "Small-flowered Cranesbill", "Sugar beet"]
cwd = os.getcwd()
dataLoc = ".\\data"
for cat in catNames:
    tv = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
    tv[catNames.index(cat)] = 1
    for i in range (1,80):
        imgLoc = str(i)+".png"
        pathImg = os.path.join("IntSys-Seedling\\data",cat,imgLoc)
        path = Path(pathImg)
        if(path.is_file()):
            #print(1)
            im_frame = cv2.imread(pathImg)
            #resizing the image to 224, 224.  This is a basic solution to the issue to varying image size
            h, w = 224, 224
            res_im = cv2.resize(im_frame, (w,h), interpolation=cv2.INTER_LINEAR)
            trainImg.append(res_im)
            trainTarget.append(tv) 
    
trainImg = np.array(trainImg)
trainTarget = np.array(trainTarget)

#now, let's train our model
history = model.fit(x=trainImg, y=trainTarget, batch_size=20, epochs = 30)


