import cv2
import torch
from ultralytics import YOLO

class ObjectDetection:

    def __init__(self, debug):
        self.debug = debug
        self.model = YOLO(model='model/yolov8n.pt')

    def get_objects(self, img):
        result = self.model(img)
        return result[0]

    def draw_boxes_cv2(self,img, result):


        # Iteriere über die erkannten Bounding Boxes und zeichne sie ein
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            x_min, y_min, x_max, y_max = box.numpy()
            label = f"{result.names[int(cls)]} {conf:.2f}"
            if conf <= 0.4:
                continue
            cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
            cv2.putText(img, label, (int(x_min), int(y_min - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
                        2)

        return img



