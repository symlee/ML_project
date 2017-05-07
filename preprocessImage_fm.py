# image preprocessing for part 2

from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
#from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from os import listdir
import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
from PIL import Image
from resizeimage import resizeimage
import time

datagen = ImageDataGenerator(
        rotation_range=360,
        width_shift_range=0,
        height_shift_range=0,
        rescale=1./255,
        shear_range=0,
        zoom_range=0,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest')


base_path = '../data/2/train_processed_small2/'
save_path = '../data/2/train_processed_small2/'
# input image dimensions
img_rows, img_cols = 390, 140
#img_rows, img_cols = 300, 110
num_aug = 20            # number of augmented pictures to create (go for 100)
num_categories = 1000   # number of unique foot prints

'''
# for resizing images and determining average image dimensions
cumuSz = np.array([[0, 0]])
counter = 0
for img_path in glob.glob("../data/2/train/*.png"):
    # for resizing images
    counter += 1
    img = Image.open(img_path)
    img_resized = img.resize((img_cols, img_rows), Image.ANTIALIAS)
    img_resized.save(save_path + str(counter) + '.png')

    # for determining average image dimensions
    #sz = np.shape(img)
    #cumuSz += sz
    #counter += 1

#avgSz = cumuSz / float(counter)
#print(avgSz)
'''


ind = 1
#ind = str(ind).zfill(5)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
for ind in xrange(1, num_categories + 1):
#for ind in xrange(1, 501):
    print('ind is: ' + str(ind))
    img = cv2.imread(base_path + str(ind) + '.png', 0)
    x = np.reshape(img, (1, img_rows, img_cols, 1))  # samples, cols, rows, channels
    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir='preview', save_prefix=str(ind), save_format='png'):
        i += 1
        if i >= num_aug:
            break  # otherwise the generator would loop indefinitely
