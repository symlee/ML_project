'''
Test on validation dataset
'''

from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.models import load_model

from os import listdir
import cv2
import numpy as np
import glob

# convert to base path and str cat
base_path = '../data/2/valid_processed_small2/'
model_file = 'v2_fm.h5'
num_chan = 1

num_images_orig = 10000
num_aug = 20
num_images_aug = num_images_orig * num_aug

# input image dimensions
img_rows, img_cols = 390, 140
input_shape = (img_rows, img_cols, num_chan)

x = np.zeros((num_images_orig, img_rows, img_cols, num_chan))

for ind in range(1, num_images_orig + 1):
    img = cv2.imread(base_path + str(ind) + '.png', 0)
    #print(img)
    x[ind - 1, :, :, :] = img
    #x_av = x_av + x[ind-1]

x = x.astype('float32')
x /= 255
print('x_train shape:', x.shape)
print(x.shape, 'train samples')
#print(x_train)


# load model and test on validation set
model = load_model(model_file)

model.summary()
import sys
#sys.exit(1)

y_predict = model.predict(x)
#print(y_predict)
y_predict = np.argmax(y_predict, axis=1)  # store col with maximum value
y_predict = y_predict + 1                 # remap labels to match assignment
np.savetxt('y_predict.csv', y_predict)
