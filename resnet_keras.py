# -*- coding: utf-8 -*-
import numpy as np # linear algebra
import random
import os # accessing directory structure
from keras.applications import ResNet50
from keras.layers import Dense, Flatten, Input, Softmax, LeakyReLU
from keras.models import Model
from keras import optimizers
from keras.initializers import RandomUniform
# GENERATING TRAINING DATA 
from PIL import Image
import cv2
from pathlib import Path
import keras.backend as kb

RAND = RandomUniform(minval=-.1, maxval=.1)
ALPHA = .01
BATCH = 30


def create_data(catCnt):
    img = []
    target = []
    catNames = ["Black-grass", "Charlock", "Cleavers", "Common Chickweed", "Common wheat", "Fat Hen", "Loose Silky-bent", "Maize", "Scentless Mayweed", "sp", "Small-flowered Cranesbill", "Sugar beet"]
    catImgCnt = [335, 454, 349, 717, 259, 544, 806, 261, 609, 277, 581, 465]
    #cwd = os.getcwd()
    #dataLoc = ".\\data"
    
    for i in range (catCnt):
        catNumber = random.randint(0,len(catNames)-1)
        catName = catNames[catNumber]
        tv = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
        tv[catNumber] = 1
        imgLoc = str(i%catImgCnt[catNumber])+".png"
        pathImg = os.path.join("IntSys-Seedling\\data",catName,imgLoc)
        path = Path(pathImg)
        if(path.is_file()):
            
            im_frame = cv2.imread(pathImg)
            if im_frame is not None:
                #resizing the image to 224, 224.  This is a basic solution to the issue to varying image size
                h, w = 224, 224
                res_im = cv2.resize(im_frame, (w,h), interpolation=cv2.INTER_LINEAR)
                img.append(res_im)
                target.append(tv) 
    return img, target

#input layer description + creation
def create_resnet_layers( in_layer, freeze):
    #inLayer = Input(shape=(224, 224, 3), name='in_layer')
    res = ResNet50(weights = 'imagenet', include_top = False)
    cntr = 0
    if freeze and (cntr < 10 or cntr > 40):
        for l in res.layers :
            l.trainable = False
        cntr += 1
    cnn = (res)(in_layer)
    flat = Flatten()(cnn)
    
    den_1 = Dense(50, kernel_initializer = RAND, name = 'den_1')(flat)
    relu_ff_1 = LeakyReLU(ALPHA, name = 'relu_ff_1')(den_1)
    den_2 = Dense(12, kernel_initializer = RAND, name = 'den_2')(relu_ff_1)
    relu_ff_2 = LeakyReLU(ALPHA, name = 'relu_ff_2')(den_2)
    output = Softmax(name = 'soft')(relu_ff_2)
    return output

#compile model
def create_model(freeze, p):
    in_layer = Input(shape=(224, 224, 3), name='in_layer')
    output = create_resnet_layers(in_layer, freeze)
    model = Model(inputs=in_layer, outputs = output)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    if(p):
        print(model.summary())
    return model

def train_baselineRes(freeze, p, cnt):
    model = create_model(freeze,p)
    img, target = create_data(cnt)
    length = len(img)
    trainImg = img[:int(length*.8)]
    trainTarget = target[:int(length*.8)]
    testImg = img[int(length*.8):]
    testTarget = target[int(length*.8):]
    model.fit(x=np.array(trainImg), y=np.array(trainTarget), batch_size=BATCH, epochs = 90, shuffle=True, validation_data = (np.array(testImg), np.array(testTarget)))
    #acc = 0
    #while acc < .9:
    #    model.fit(x=np.array(trainImg), y=np.array(trainTarget), batch_size=BATCH, epochs = 100, shuffle=True, validation_data = (np.array(testImg), np.array(testTarget)))
    #    loss, acc = model.evaluate(x=np.array(testImg), y=np.array(testTarget), verbose=1)
    #print(loss, "\n", acc)
    model.save_weights('resBaseLine.h5')
        
    


