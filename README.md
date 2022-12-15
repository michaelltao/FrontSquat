# FrontSquat
## Overview
Neural network implentation via TensorFlow for depth detection in a powerlifting squat trained over a squat dataset sourced from Kaggle. The training dataset consists exclusively of squats from the front angle, and thus the model is only applicable to videos from the same angle. 
>
## Implementation   
- Sourced data processed in `processData.py` which concatenates the subject's keypoints in the frames of each video, resulting in one large array per video. The resulting arrays were then used to train the model. 
- Google's MediaPipe pose estimation solution used to detect subject's keypoints throughout the video. 
- Input video is parsed via `getKeypoints.py` using MediaPipe and output coordinates concatenated into a numpy array for input to the neural network. 
### Parameters and Training
- Train-test split of 0.8 and 0.2
- Model trained with batch size of 32 and 200 epochs
- 3 layers with 32, 64, and 1 neurons where the first two layer activation functions are ReLU and the last layer is a sigmoid
>
## Installation and Example  
Install requirements `pip install -r requirements.txt` 

Compile the model by running `model.py`

Run `demo.py` for a demonstration of how the model works with an input  
- Function takes in a path to a video (provided), runs it through the trained network, and prints whether squat depth was achieved

## References
Google's MediaPipe [pose estimatation](https://google.github.io/mediapipe/solutions/pose.html) solution to obtain landmark coordinates and [Kaggle Powerlifting](https://www.kaggle.com/datasets/ayoobaboosalih/powerlifting-squat-dataset)

