from processData import processData
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Flatten
import numpy as np
from sklearn.model_selection import train_test_split

x = np.array(processData())
y = np.array([0] * 120 + [1] * 120)

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=.2)


model = Sequential()
model.add(Flatten())
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics='accuracy')


model.fit(x, y, epochs=200, batch_size=32)

model.save('FrontSquatNN')





