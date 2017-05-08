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


# convert to base path and str cat
num_images_total = 10000  # total number of images (including left and right)
num_images_ind = num_images_total/2
base_path = '../data/1/test_processed_small/'
model_file = 'v1.h5'

# input image dimensions
#img_rows, img_cols = 470, 230
img_rows, img_cols = 235, 115
input_shape = (img_rows, img_cols, 1)

x = np.zeros((num_images_total, img_rows, img_cols, 1))

#x_av = np.zeros((img_rows, img_cols, 1))
for ind in range(1, num_images_total + 1):
    img = cv2.imread(base_path + str(ind) + '.png', 0)
    #print(img)
    x[ind - 1, :, :, 0] = img
    #x_av = x_av + x[ind-1]

#print("x_Av max = " + str(x_av.max()/num_images_total))

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

