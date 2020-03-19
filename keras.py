# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 15:40:00 2020

@author: User
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

n_classes=10
epochs=10
batch_size = 32
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

train_labels_ohe = tf.one_hot(train_labels, depth=n_classes).numpy()
test_labels_ohe = tf.one_hot(test_labels, depth=n_classes).numpy()


model = tf.keras.models.Sequential([
 tf.keras.layers.Flatten(),
 tf.keras.layers.Dense(512,activation=tf.nn.relu),
 tf.keras.layers.Dropout(0.2),
 tf.keras.layers.Dense(10,activation=tf.nn.softmax)
])
optimiser = tf.keras.optimizers.Adam()
model.compile (optimizer= optimiser,
loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(train_labels_ohe, train_labels_ohe,batch_size=batch_size, epochs=10)
model.evaluate(test_images, test_labels)
model.summary()
model.save_weights(r'C:\Users\User\Documents\weights')


