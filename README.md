# TFLitePoseEstimation

This code snipset is heavily based on <b><a href="https://www.tensorflow.org/lite/examples/pose_estimation/overview">TensorFlow Lite Pose Estimation</a></b><br>
The detection model can be downloaded from above link.<br>
For the realtime implementation on Android look into the <a href="https://github.com/tensorflow/examples/tree/master/lite/examples/posenet/android">Android Pose Estimation Example</a><br>
Follow the <a href="https://github.com/joonb14/TFLitePoseEstimation/blob/main/pose%20estimation.ipynb">pose estimation.ipynb</a> to get information about how to use the TFLite model in your Python environment.<br>

### Details
The <b>posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite</b> file's input takes normalized 257x257x3 shape image. And the output is composed of 4 different outputs. The 1st output contains the heatmaps, 2nd output contains the offsets, 3rd output contains the forward_displacements, 4th output contains the backward_displacements.<br>

For model inference, we need to load, resize, typecast the image.<br>
In my case for convenience used pillow library to load and just applied /255 for all values then cast the numpy array to float32.<br>
<img src="https://user-images.githubusercontent.com/30307587/110313718-2f093580-804a-11eb-8961-0d67383be16e.png" width=400px/><br>
Then if you follow the correct instruction provided by Google in <a href="https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_python">load_and_run_a_model_in_python</a>, you would get output in below shape<br>
<img src="https://user-images.githubusercontent.com/30307587/110313834-5cee7a00-804a-11eb-8182-943423d0c6c2.png" width=600px/><br>
Now we need to process this output to use it for pose estimation<br>

##### Extract Key points
```python
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))
    
height = heatmaps[0].shape[0]
width = heatmaps[0][0].shape[0]
numKeypoints = heatmaps[0][0][0].shape[0]

keypointPositions = []

for keypoint in range(numKeypoints):
    maxVal = heatmaps[0][0][0][keypoint]
    maxRow = 0
    maxCol = 0
    for row in range(height):
        for col in range(width):
            if (heatmaps[0][row][col][keypoint] > maxVal):
                maxVal = heatmaps[0][row][col][keypoint]
                maxRow = row
                maxCol = col
    keypointPositions.append([maxRow,maxCol])


confidenceScores=[]
yCoords = []
xCoords = []
for idx, position in enumerate(keypointPositions):
    positionY = keypointPositions[idx][0]
    positionX = keypointPositions[idx][1]
    yCoords.append(position[0] / (height - 1) * 257 + offsets[0][positionY][positionX][idx])
    xCoords.append(position[1] / (width - 1) * 257 + offsets[0][positionY][positionX][idx + numKeypoints])
    confidenceScores.append(sigmoid(heatmaps[0][positionY][positionX][idx]))
#     yCoords.append()
score = np.average(confidenceScores)
score
```

##### Visualize Key points and Body joints
```python
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


class BodyPart(Enum):
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR= 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
  
bodyJoints = np.array(
    [(BodyPart.LEFT_WRIST, BodyPart.LEFT_ELBOW),
    (BodyPart.LEFT_ELBOW, BodyPart.LEFT_SHOULDER),
    (BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER),
    (BodyPart.RIGHT_SHOULDER, BodyPart.RIGHT_ELBOW),
    (BodyPart.RIGHT_ELBOW, BodyPart.RIGHT_WRIST),
    (BodyPart.LEFT_SHOULDER, BodyPart.LEFT_HIP),
    (BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP),
    (BodyPart.RIGHT_HIP, BodyPart.RIGHT_SHOULDER),
    (BodyPart.LEFT_HIP, BodyPart.LEFT_KNEE),
    (BodyPart.LEFT_KNEE, BodyPart.LEFT_ANKLE),
    (BodyPart.RIGHT_HIP, BodyPart.RIGHT_KNEE),
    (BodyPart.RIGHT_KNEE, BodyPart.RIGHT_ANKLE)]
)

minConfidence = 0.5

fig, ax = plt.subplots(figsize=(10,10))

if (score > minConfidence):
    ax.imshow(res_im)
    for line in bodyJoints:
        plt.plot([xCoords[line[0].value],xCoords[line[1].value]],[yCoords[line[0].value],yCoords[line[1].value]],'k-')
    ax.scatter(xCoords, yCoords, s=30,color='r')
    plt.show()
```

<img src="https://user-images.githubusercontent.com/30307587/110314123-cd959680-804a-11eb-84f2-198a45a50618.png" width=600px/><br>
I believe you can modify the rest of the code as you want by yourself.<br>
Thank you!<br>