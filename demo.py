import tensorflow as tf
from getKeypoints import getKeypoints


def isDepth(videoPath):
    model = tf.keras.models.load_model('FrontSquatNN')

    testData = getKeypoints(videoPath)
    modelIn = [testData, testData]

    prediction = model.predict(modelIn)
    result = (prediction[0] + prediction[1]) / 2

    if result > 0.5:
        print('Depth!')
    else:
        print('No Depth!')

    return result


print(isDepth('/Users/michaeltao/PycharmProjects/FrontFaceSquat/Videos/IMG_1843.MOV'))

