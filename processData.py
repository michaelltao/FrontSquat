import numpy as np
import os


def processData():
    invalid = []
    for i in range(120):
        vidCoords = []
        counter = 0
        while True:
            try:
                frame = np.load(os.path.join('Squat_Data/Invalid', '{}'.format(i), "{}.npy".format(counter)))
                counter += 1
                for keypoint in range(0, len(frame), 4):
                    vidCoords.append((frame[keypoint], frame[keypoint+1]))
            except:
                break
        invalid.append(vidCoords)

    valid = []
    for i in range(120):
        vidCoords = []
        counter = 0
        while True:
            try:
                frame = np.load(os.path.join('Squat_Data/Valid', '{}'.format(i), "{}.npy".format(counter)))
                counter += 1
                for keypoint in range(0, len(frame), 4):
                    vidCoords.append((frame[keypoint], frame[keypoint+1]))

            except:
                break
        valid.append(vidCoords)

    return valid + invalid

