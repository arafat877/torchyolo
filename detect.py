import cv2
import torch
from torchvision.transforms import transforms

from torchyolo import models
import numpy as np
from lsnms import nms

from torchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS
from torchyolo.utils.utils import load_classes, xywh2xyxy, xywh2xyxy_np, non_max_suppression, rescale_boxes

img_size, conf_thres, nms_thres = 416, 0.5, 0.5
model = models.load_model("cfg/yolov3-tiny.cfg", "weights/yolov3-tiny.weights")
category_names = load_classes("cfg/coco.names")
image = cv2.imread("tests/data/street.jpg")

model.eval()

input_img = transforms.Compose([
    DEFAULT_TRANSFORMS,
    Resize(img_size)])(
    (image, np.zeros((1, 5))))[0].unsqueeze(0)


# Get detections
with torch.no_grad():
    detections = model(input_img)
    detections_nms = non_max_suppression(detections, conf_thres, nms_thres)
    print("detections_nms", detections_nms)
    detections_lsnms = nms(detections, conf_thres, nms_thres)
    print("detections_lsnms", detections_lsnms)

"""
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
predictions = detect(model, img)
boxes = predictions[:, :4]
scores = predictions[:, 4]
class_ids = predictions[:, 5].astype(np.int32)
keep = nms(boxes, scores, iou_threshold=0.5, class_ids=class_ids)

"""
"""
def detect(model, image, img_size=416, conf_thres=0.5, nms_thres=0.5):
    model.eval()

    # Configure input
    input_img = transforms.Compose([
        DEFAULT_TRANSFORMS,
        Resize(img_size)])(
        (image, np.zeros((1, 5))))[0].unsqueeze(0)

    if torch.cuda.is_available():
        input_img = input_img.to("cuda")

    # Get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = non_max_suppression(detections, conf_thres, nms_thres)
        detections = rescale_boxes(detections[0], img_size, image.shape[:2])
    return detections.numpy()


model = models.load_model("cfg/yolov3-tiny.cfg", "weights/yolov3-tiny.weights")
img = cv2.imread("tests/data/street.jpg")
category_names = load_classes("cfg/coco.names")

# data preprocessing for detection
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
predictions = detect(model, img)
print(predictions)

"""
