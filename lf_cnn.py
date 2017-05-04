'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from os import listdir
import cv2
import numpy as np

# parameters
batch_size = 128
num_classes = 2
epochs = 12
train_val_ratio = 0.5

# convert to base path and str cat
#num_images = 10000  # total number of images
num_images_total = 8  # total number of images (including left and right)
num_images_ind = num_images_total/2
base_path = './data/1/train/testing/'

# input image dimensions
#img_rows, img_cols = 28, 28
img_rows, img_cols = 470, 230

x = np.zeros((num_images_total, img_rows, img_cols, 1))
y = np.zeros((num_images_total, img_rows, img_cols, 1))

for ind in range(1, num_images_ind + 1):
    img_left = cv2.imread(base_path + 'left/' + str(ind) + '.png', 0)
    img_right = cv2.imread(base_path + 'right/' + str(ind) + '.png', 0)
    x[ind - 1, :, :, 0] = img_left
    x[ind + num_images_ind - 1, :, :, 0] = img_right

y = np.concatenate((np.ones(num_images_ind), np.ones(num_images_ind)*2))

# the data, shuffled and split between train and test sets
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

# make train and test datasets through random permutation (verified with 4 images)
assert len(x) == len(y)
p = np.random.permutation(len(y))
x = x[p]
y = y[p]

train_val_split = int(np.ceil(train_val_ratio * num_images_total))
x_train = x[0:train_val_split]
y_train = y[0:train_val_split]
x_test = x[train_val_split:]
y_test = y[train_val_split:]

#im = cv2.imread('./data/1/train/processed_small/left/1.png', 0)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])