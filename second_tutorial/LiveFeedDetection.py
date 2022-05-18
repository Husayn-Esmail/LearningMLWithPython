#!../mlvenv/bin/python3

# import modules
import os
from imageai.Detection import VideoObjectDetection
import cv2

camera = cv2.VideoCapture(0)

exec_path = os.getcwd()

detector = VideoObjectDetection()
detector.setModelTypeAsRetinaNet()
# set path of the model
detector.setModelPath("resnet50_coco_best_v2.1.0.h5")
# loads the model from the path specified
detector.loadModel()

# used to detect from camera input

video_path = detector.detectObjectsFromVideo(camera_input=camera,
output_file_path=os.path.join(exec_path, "camera_detected_video"),frames_per_second=20, log_progress=True,minimum_percentage_probability=30)

print(video_path)

