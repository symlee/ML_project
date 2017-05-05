# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Convolution2D
from keras import backend as K
import numpy as np
import cv2
batch_size = 32
num_classes = 2
epochs = 12

# input image dimensions
img_rows, img_cols = 350, 160

# Number of samples
N = 10000
Nl = 5000
y_train = np.concatenate( ( np.ones(Nl) , np.zeros(Nl) ) )
x_train = np.zeros( (N, img_rows, img_cols, 1) )
input_shape = (img_rows, img_cols, 1)

# Read in image files
base_path = '../data/1/train/small2/'
for idx in range(1, Nl+1):
    img_left = cv2.imread(base_path+'left/'+str(idx)+'.png', cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread(base_path+'right/'+str(idx)+'.png', cv2.IMREAD_GRAYSCALE)
    #print(img_left)
    #print(img_left)
    x_train[idx-1, :, :, 0] = img_left
    x_train[idx+Nl-1, :, :, 0] = img_right

x_train = x_train.astype('float32')
x_train /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
y_train.shape
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


#model.compile(loss=keras.losses.categorical_crossentropy,
model.compile(loss=keras.losses.categorical_crossentropy,
               optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.summary()
#import sys
#sys.exit(1)

B = N
intervals = [ (ix, ix + B) for ix in range(0, N, B)]
intervals[-1] = (intervals[-1][0], N)
for e in range(0, epochs):
    # Shuffle data
    idx = np.arange(N)
    np.random.shuffle(idx)
    x_train = x_train[idx]
    y_train = y_train[idx]
    for k in intervals:
        model.fit(x_train[k[0]:k[1], :, :],
                      y_train[k[0]:k[1], :],
                      batch_size=batch_size,
                      epochs=1,
                      verbose=1)

# Read in validation data
#Nv = 10000
#test_data_path = '/home/cormac/classes/ML/proc/1/valid/'
#x_test = np.zeros( (Nv, img_rows, img_cols, 1))
#for idx in range(1, Nv+1):
#    x_test[idx-1, :, :, 0] = cv2.imread(test_data_path+str(idx)+'.png',
#                                            cv2.IMREAD_GRAYSCALE)
#
#x_test /= 255
#
## Predict labels
#y_test = model.predict(x_test)
#labels = np.argmax(y_test, axis=1)

# Write output to file
#f = open('/home/cormac/classes/ML/proc/1/1.csv', 'w')
#for v in labels.tolist():
#    f.write(str(v+1) +'\n')
#f.close()

#score = model.evaluate(x_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
