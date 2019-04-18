import numpy as np # linear algebra
import os # accessing directory structure
import skimage

import imageio

from shutil import copyfile
from sklearn.model_selection import train_test_split

from PIL import Image
import cv2

import pickle


# GENERATING TRAINING DATA

def generateData(img_size):
    """
    @param: img_size: size of image
    
    prints failed image paths
    
    @return: trainImg, trainTarget, validImg, validTarget
    """
    allImg = []
    allTarget = []

    catNames = ["Black-grass", "Charlock", "Cleavers", "Common Chickweed", "Common wheat", "Fat Hen", "Loose Silky-bent", "Maize", "Scentless Mayweed", "Shepherds Purse", "Small-flowered Cranesbill", "Sugar beet"]

    for cat in catNames:
        tv = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
        tv[catNames.index(cat)] = 1
        assert np.max(tv) == 1 # make sure properly classfifiedd
        for i in range (0,max_balanced):
            imgPath = os.path.join(balanced_dir, cat + "_" + str(i) + ".png")
            if(os.path.isfile(imgPath)):
                im_frame = cv2.imread(imgPath)
                #resizing the image to img_size, img_size.  This is a basic solution to the issue to varying image size
                res_im = cv2.resize(im_frame, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
                allImg.append(res_im)
                allTarget.append(tv)
            else:
                print(imgPath)

    allImg = np.array(allImg)
    allTarget = np.array(allTarget)
    
    # train valid split
    stratify = np.argmax(allTarget, axis = 1).reshape((allTarget.shape[0], 1))
    return train_test_split(allImg, allTarget, test_size=0.1, random_state=13, stratify=allTarget)


def main():
	# find max balanced

	data_path = os.path.join(".", "data")
	dir_list = os.listdir(data_path)

	max_balanced = 9999999999999

	for dir_ in dir_list:
	    n = 0
	    print(dir_, end=': ')
	    for name in os.listdir(os.path.join(data_path, dir_)):
	        if os.path.isfile(os.path.join(data_path, dir_, name)):
	            n = n + 1
	    print(n)
	    max_balanced = min(max_balanced, n)
	print("---------")
	print("Max Balanced:", max_balanced)

	# create balanced dataset if doest exist already


	balanced_dir = os.path.join(".", 'balanced')

	if not os.path.exists(balanced_dir):
	    print("creating new balanced dataset with", max_balanced, "imgs per class")
	    os.mkdir(balanced_dir)

	    for dir_ in dir_list:
	        n = 0
	        for name in os.listdir(os.path.join(data_path, dir_)):
	            src = os.path.join(data_path, dir_, name)
	            if os.path.isfile(src):
	                if(n < max_balanced):
	                    dst = os.path.join(balanced_dir, dir_ + "_" + str(n) + ".png")
	                    copyfile(src, dst)
	                    n = n + 1
	    print("finished creating new balanced dataset")
	else:
	    print("balanced dataset already exists")
	print(len(os.listdir(balanced_dir)))
	print(12 * 253)

	img_size = 128
	trainX, validX, trainY, validY = generateData(img_size=img_size)

	# pickle load
	pickle_dir = os.path.join(".", 'balanced_pickled')

	if not os.path.exists(pickle_dir):
	    os.mkdir(pickle_dir)

	with open(os.path.join(".", "balanced_pickled", "trainX_" + str(img_size)), "wb") as f:
	    pickle.dump(trainX, f)
	with open(os.path.join(".", "balanced_pickled", "trainY_" + str(img_size)), "wb") as f:
	    pickle.dump(trainY, f)
	with open(os.path.join(".", "balanced_pickled", "validX_" + str(img_size)), "wb") as f:
	    pickle.dump(validX, f)
	with open(os.path.join(".", "balanced_pickled", "validY_" + str(img_size)), "wb") as f:
	    pickle.dump(validY, f)

	dataset = (trainX, trainY, validX, validY)
	with open(os.path.join(".", "balanced_pickled", "dataset_" + str(img_size)), "wb") as f:
    	pickle.dump(dataset, f)
if __name__=='__main__':
    main()