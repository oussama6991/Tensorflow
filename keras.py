# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 15:40:00 2020

@author: User
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
print(tf.keras.__version__)


(train_x,train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
epochs=10
batch_size = 32

train_x, test_x = tf.cast(train_x/255.0, tf.float32), tf.cast(test_x/255.0,tf.float32)
train_y, test_y = tf.cast(train_y,tf.int64),tf.cast(test_y,tf.int64)


model = tf.keras.models.Sequential([
 tf.keras.layers.Flatten(),
 tf.keras.layers.Dense(512,activation=tf.nn.relu),
 tf.keras.layers.Dropout(0.2),
 tf.keras.layers.Dense(10,activation=tf.nn.softmax)
])
optimiser = tf.keras.optimizers.Adam()
model.compile (optimizer= optimiser,
loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs)
model.evaluate(test_x, test_y)
model.summary()
model.save('./model')

new_model = load_model('./model')

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
model2 = tf.keras.models.Sequential([
 tf.keras.layers.Flatten(),
 tf.keras.layers.Dense(512,activation=tf.nn.relu),
 tf.keras.layers.Dropout(0.2),
 tf.keras.layers.Dense(10,activation=tf.nn.softmax)
])
optimiser2 = tf.keras.optimizers.Adam()
model2.compile (optimizer= optimiser,
loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
model2.fit(train_x, train_y, batch_size=batch_size, epochs=epochs)
model2.evaluate(test_x, test_y)