from processData import processData
from getKeypoints import getKeypoints
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Flatten
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x = np.array(processData())
y = np.array([1] * 120 + [0] * 120)

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=.2)


model = Sequential()
model.add(Flatten())
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics='accuracy')


model.fit(X_train, Y_train, epochs=200, batch_size=32)

model.save('FrontSquatNN')





