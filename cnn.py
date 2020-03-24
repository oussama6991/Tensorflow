# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 01:31:55 2020

@author: User
"""

import tensorflow as tf

from tensorflow.keras import datasets, layers, models

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential,Input,Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers.advanced_activations import LeakyReLU

batch_size = 64
epochs = 20
num_classes = 10

fashion_mnist = keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_test = x_test / 255.0
x_train = x_train / 255.0

model = Sequential()
model.add(Conv2D(32, kernel_size=(5,5), input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(5,5), input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) 
model.add(Dense(1024, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))

model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
model.summary()
model.fit(x=x_train,y=y_train, epochs=10)

model.save('trained_model')
model = keras.models.load_model('trained_model')
