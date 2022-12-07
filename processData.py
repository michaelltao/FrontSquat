import numpy as np
import os


def processData():
    maxCount = 0
    invalid = []
    for i in range(120):
        vidCoords = []
        counter = 0
        while counter < 192:
            try:
                frame = np.load(os.path.join('Squat_Data/Invalid', '{}'.format(i), "{}.npy".format(counter)))
                counter += 1
                x, y = frame[::4], frame[1::4]
                coords = list(zip(x, y))
                vidCoords.append(coords)
            except:
                # break
                counter += 1
                nonVal = (0, 0)
                coords = [tuple(nonVal) for _ in range(33)]
                vidCoords.append(coords)
        invalid.append(vidCoords)
        maxCount = max(counter, maxCount)

    valid = []
    for i in range(120):
        vidCoords = []
        counter = 0
        while counter < 192:
            try:
                frame = np.load(os.path.join('Squat_Data/Valid', '{}'.format(i), "{}.npy".format(counter)))
                counter += 1
                x, y = frame[::4], frame[1::4]
                coords = list(zip(x, y))
                vidCoords.append(coords)
            except:
                # break
                counter += 1
                nonVal = (0, 0)
                coords = [tuple(nonVal) for _ in range(33)]
                vidCoords.append(coords)
        valid.append(vidCoords)

    return invalid + valid


