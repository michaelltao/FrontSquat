from processData import processData
from getKeypoints import getKeypoints
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense
import numpy as np

x_train = processData()
X_train = tf.ragged.constant(x_train)
Y_train = tf.convert_to_tensor([1] * 120 + [0] * 120)

# X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
# Y_train = tf.convert_to_tensor(Y_train, dtype=tf.float32)

# print(type(X_train))

# model = Sequential()
# model.add(Dense(units=32, activation='relu'))
# model.add(Dense(units=64, activation='relu'))
# model.add(Dense(units=1, activation='sigmoid'))
#
# model.compile(loss='binary_crossentropy', optimizer='sgd', metrics='accuracy')

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=[None,1], dtype=tf.int64, ragged=True),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.SGD(), metrics='accuracy')

model.fit(X_train, Y_train, epochs=200, batch_size=32)


