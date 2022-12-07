import tensorflow as tf
from getKeypoints import getKeypoints

model = tf.keras.models.load_model('FrontSquatNN')

# print(model.summary())
testData = getKeypoints('Videos/test.MOV').reshape((12672,))

prediction = model.predict(testData)