import cv2
import mediapipe as mp
import numpy as np


def getKeypoints(videoPath):
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(videoPath)
    allXY = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = [i * (total_frames // 192) for i in range(192)]

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()

            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            for j in range(33):
                allXY.append(results.pose_landmarks.landmark[j].x)
                allXY.append(results.pose_landmarks.landmark[j].y)

    cap.release()
    return np.array(allXY)


# path2Vid = 'Videos/test.MOV'
# arr = getKeypoints(path2Vid)
# print(arr.shape)

