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

datagen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False,
        fill_mode='nearest')

'''
datagen = ImageDataGenerator(
        rotation_range=0,
        width_shift_range=0,
        height_shift_range=0,
        rescale=1./255,
        shear_range=0,
        zoom_range=0,
        horizontal_flip=False,
        fill_mode='nearest')
'''

base_path = '../data/2/train/'

# input image dimensions
#img_rows, img_cols = 470, 230
img_rows, img_cols = 235, 115
input_shape = (img_rows, img_cols, 1)

ind = 1
ind = str(ind).zfill(5)
img_right = cv2.imread(base_path + str(ind) + '.png', 0)
#plt.imshow(img_right, cmap='gray', interpolation='bicubic')
#plt.show()
x = np.reshape(img_right, (1, img_rows, img_cols, 1)) # samples, cols, rows, channels

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='preview', save_prefix=str(ind), save_format='png'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely
