from torchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS
from torchyolo.utils.utils import load_classes
from torchvision.transforms import Compose
from torchyolo import models
from lsnms import nms
import numpy as np
import torch, cv2


def lsnms_detect(nms_thres):
    input_img = Compose([
        DEFAULT_TRANSFORMS,
        Resize(img_size)])(
        (image, np.zeros((1, 5))))[0].unsqueeze(0)

    # Get detections
    with torch.no_grad():
        detections = model(input_img)
        boxes = detections[:, :4]
        scores = detections[:, 4]
        class_ids = detections[:, 5].astype(np.int32)
        detections = nms(boxes, scores, nms_thres, class_ids)
        print("detections: ", detections)


if __name__ == "__main__":
    img_size, conf_thres, nms_thres = 416, 0.5, 0.5
    model = models.load_model("cfg/yolov3-tiny.cfg", "weights/yolov3-tiny.weights")
    category_names = load_classes("cfg/coco.names")
    image = cv2.imread("tests/data/street.jpg")