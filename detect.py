# Library imports

import cv2
from torchyolo import models
import numpy as np
from lsnms import nms
from torchyolo.utils.utils import load_classes, xywh2xyxy, xywh2xyxy_np
from torchyolo.predict import detect_image

# model,img and category_names file path

model = models.load_model("cfg/yolov3-tiny.cfg", "weights/yolov3-tiny.weights")
img = cv2.imread("tests/data/street.jpg")
category_names = load_classes("cfg/coco.names")

# data preprocessing for detection
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
predictions = detect_image(model, img)
boxes = predictions[:, :4]
scores = predictions[:, 4]
class_ids = predictions[:, 5].astype(np.int32)

# apply non-maximum suppression
keep = nms(boxes, scores, iou_threshold=0.5, class_ids=class_ids)



