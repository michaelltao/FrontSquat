import cv2
import mediapipe as mp
import numpy as np


def getKeypoints(videoPath):
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(videoPath)
    allX, allY = [], []

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            try:
                ret, frame = cap.read()

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                for i in range(33):
                    allX.append(results.pose_landmarks.landmark[i].x)
                    allY.append(results.pose_landmarks.landmark[i].y)
            except:
                break
    cap.release()
    return list(zip(allX, allY))


# path2Vid = 'Videos/test.MOV'
# print(len(getKeypoints(path2Vid)))
