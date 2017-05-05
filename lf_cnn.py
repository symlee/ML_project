'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.applications.vgg16 import VGG16

from os import listdir
import cv2
import numpy as np

# parameters
batch_size = 64
num_classes = 2
epochs = 12

train_val_ratio = 1 # percentage used for training
n = 30   # number of images per chunk (current works at around 100, crashes at 200)

# convert to base path and str cat
num_images_total = 10000  # total number of images (including left and right)
num_images_ind = num_images_total/2
base_path = '../data/1/train/processed_small/'

# input image dimensions
#img_rows, img_cols = 470, 230
img_rows, img_cols = 235, 115

x = np.zeros((num_images_total, img_rows, img_cols, 1))
y = np.zeros((num_images_total, img_rows, img_cols, 1))

for ind in range(1, num_images_ind + 1):
    img_left = cv2.imread(base_path + 'left/' + str(ind) + '.png', 0)
    img_right = cv2.imread(base_path + 'right/' + str(ind) + '.png', 0)
    #print(img_left)
    #print(img_right)
    x[ind - 1, :, :, 0] = img_left
    x[ind + num_images_ind - 1, :, :, 0] = img_right

y = np.concatenate((np.zeros(num_images_ind), np.ones(num_images_ind)))

# make train and validation datasets through random permutation (verified with 4 images)
p = np.random.permutation(len(y))
x = x[p]
y = y[p]

train_val_split = int(np.ceil(train_val_ratio * num_images_total))
x_train = x[0:train_val_split]
y_train = y[0:train_val_split]
x_test = x[train_val_split:]
y_test = y[train_val_split:]

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
print(x_train)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


#mnist
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


'''
#building powerful image classification models using very little data
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))
'''

#model = VGG16

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adagrad(),
              metrics=['accuracy'])



model.summary()
import sys
#sys.exit(1)

'''
# chunks for
x_train_chunks = [x_train[i:i+n] for i in xrange(0, len(x_train), n)]
y_train_chunks = [y_train[i:i+n] for i in xrange(0, len(y_train), n)]


for epoch in xrange(0, 12):
    for ind in xrange(0, len(x_train_chunks)):
        model.fit(x_train_chunks[ind], y_train_chunks[ind], batch_size=len(x_train_chunks[ind]), epochs=1, verbose=1, validation_data=(x_test, y_test))
'''
model.fit(x_train, y_train, batch_size=batch_size, epochs=12, verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
