from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.models import Model

from os import listdir
import cv2
import numpy as np
import glob

# parameters
batch_size = 50
num_classes = 1000
epochs = 12
train_val_ratio = 0.7 # percentage used for training

num_images_orig = 10   # TODO - put this back at 1000
num_aug = 20
num_images_aug = num_images_orig * num_aug
base_path = '../data/2/train_processed_small_aug2/'

img_rows, img_cols = 390, 140
input_shape = (img_rows, img_cols, 3)

train_val_split_dataset = int(np.ceil(train_val_ratio * num_images_aug))
train_val_split_class = int(train_val_ratio * num_aug)

x_train = np.zeros((train_val_split_dataset, img_rows, img_cols, 3))
y_train = np.zeros(train_val_split_dataset)
x_test = np.zeros((num_images_aug - train_val_split_dataset, img_rows, img_cols, 3))
y_test = np.zeros(num_images_aug - train_val_split_dataset)

for orig_ind in range(1, num_images_orig + 1):
    aug_list = glob.glob(base_path + str(orig_ind) + '_*')
    for aug_ind in range(0, len(aug_list)):
        img = cv2.imread(aug_list[aug_ind], 1)
        #print(img)
        if aug_ind < train_val_split_class:
            x_train[(orig_ind - 1) * train_val_split_class + aug_ind, :, :, :] = img
            y_train[(orig_ind - 1) * train_val_split_class + aug_ind] = orig_ind - 1
        else:
            x_test[(orig_ind - 1) * (num_aug - train_val_split_class) + (aug_ind - train_val_split_class), :, :, :] = img
            y_test[(orig_ind - 1) * (num_aug - train_val_split_class) + (aug_ind - train_val_split_class)] = orig_ind - 1


# make train and validation datasets through random permutation (verified with 4 images)
p_train = np.random.permutation(len(y_train))
p_test = np.random.permutation(len(y_test))

x_train = x_train[p_train]
y_train = y_train[p_train]
x_test = x_test[p_test]
y_test = y_test[p_test]

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape, 'train samples')
print(x_test.shape[0], 'test samples')
#print(x_train)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# include_top needs to be false to be able to specify input_shape
base_model = keras.applications.inception_v3.InceptionV3(include_top=False, weights=None, input_tensor=None, input_shape=input_shape)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


model.summary()
import sys
#sys.exit(1)

early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(x_train, y_train, batch_size=batch_size, epochs=12, verbose=1, validation_data=(x_test, y_test), callbacks=[early_stopping])
model.save('v3_fm.h5')

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])