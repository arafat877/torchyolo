import cv2
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchyolo.models import load_model
from torchyolo.utils.datasets import  ImageFolder
from torchyolo.utils.transforms import DEFAULT_TRANSFORMS, Resize
from torchyolo.utils.utils import non_max_suppression, load_classes


def create_data_loader():
    dataset = ImageFolder(
        img_path,
        transform=transforms.Compose([DEFAULT_TRANSFORMS, Resize(img_size)]))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu,
        pin_memory=True)
    return dataloader


def run():
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # data loader
    dataloader = create_data_loader()
    # model
    model = load_model(model_path, weights_path)
    model.to(device).eval()
    # predict
    for i, (img, _) in enumerate(dataloader):
        with torch.no_grad():
            output = model(img.to(device))
            output = non_max_suppression(output, conf_thres)
        print(output)
        img = img.to('cpu').numpy().transpose(1, 2, 0)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (416, 416))
        cv2.imwrite("deneme.jpg", img)
        print(f"---- Detections were saved to: '{output_path}' ----")


if __name__ == "__main__":
    model_path = "../cfg/yolov3-tiny.cfg"
    weights_path = "../weights/yolov3-tiny.weights"
    img_path = "../tests/data/dog.jpg"
    output_path = "../output"
    batch_size, img_size, n_cpu, conf_thres, nms_thres = 1, 416, 8, 0.5, 0.5
    classes = load_classes("../cfg/coco.names")
    run()
