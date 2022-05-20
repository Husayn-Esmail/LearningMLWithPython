#!../mlvenv/bin/python3

# import modules
from imageai.Detection import ObjectDetection
import os
# gets the current path of the py file to execute commands in
execution_path = os.getcwd()

# starts the detection
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path, "resnet50_coco_best_v2.1.0.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path ,"image2.jpeg"), output_image_path=os.path.join(execution_path, "imagenew.jpg"))

# prints respective objects and their accuracy
#for eachObject in detections:
#	print(eachObject["name"], ":", eachObject["percentage_probability"])
for object in detections:
	if (object["name"] == "cat"):
		print("A cat was detected");
	else:
		print("no cat was detected")

