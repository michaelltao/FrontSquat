# FrontSquat

Neural network implentation via TensorFlow for depth detection in a powerlifting squat trained over a squat dataset sourced from Kaggle. The training dataset consists exclusively of squats from the front angle, and thus the model is only applicable to videos from the same angle. Sourced data processed in `processData.py` which concatenates the subject's keypoints in the frames of each video, resulting in one large array per video. The resulting arrays were then used to train the model. 
>

Google's MediaPipe pose estimation solution used to detect subject's keypoints throughout the video. Input video is parsed via `getKeypoints.py` using MediaPipe and output coordinates concatenated into a numpy array for input to the neural network. 

> 
Run `demo.py` for a demonstration of how the model works  
- Function takes in a path to a video (provided), runs it through the trained network, and prints whether squat depth was achieved

