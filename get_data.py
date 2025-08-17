from roboflow import Roboflow
import supervision as sv
import cv2

rf = Roboflow(api_key="olDraIOU4RlnvbRkLtju")
project = rf.workspace().project("traffic-signs-detection-europe")
model = project.version(11).model
datasets = project.version(11).download("yolov8")