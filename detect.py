import cv2
from torchyolo import models
import numpy as np
from lsnms import nms
from torchyolo.utils.utils import load_classes, xywh2xyxy, xywh2xyxy_np
from torchyolo.predict import detect_image

model = models.load_model("cfg/yolov3-tiny.cfg", "weights/yolov3-tiny.weights")
img = cv2.imread("tests/data/street.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
category_names = load_classes("cfg/coco.names")
category_mapping = {i: category_names[i] for i in range(len(category_names))}
# process predictions
predictions = detect_image(model, img)
boxes = predictions[:, :4]
boxes = np.array(boxes).tolist()
scores = predictions[:, 4]
scores = np.array(scores).tolist()
class_ids = predictions[:, 5]
class_ids = np.array(class_ids).tolist()
class_ids = [int(i) for i in class_ids]

# apply non-maximum suppression
keep = nms(boxes, scores, iou_threshold=0.5, class_ids=class_ids)




