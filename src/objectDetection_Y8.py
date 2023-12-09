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

    def draw_boxes_cv2(self,image, result):
        # Überprüfe, ob das Bild und die Ergebnisse vorhanden sind
        if image is None or result is None:
            print("Bild oder Ergebnisse fehlen.")
            return

        # Konvertiere das Bild von Tensor zu NumPy (falls es ein Tensor ist)
        if torch.is_tensor(image):
            image = image.permute(1, 2, 0).cpu().numpy()

        # Kopiere das Bild für die Anzeige
        image_display = image.copy()

        # Iteriere über die erkannten Bounding Boxes und zeichne sie ein
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            x_min, y_min, x_max, y_max = box.numpy()
            label = f"{result.names[int(cls)]} {conf:.2f}"
            if conf <= 0.4:
                continue
            cv2.rectangle(image_display, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
            cv2.putText(image_display, label, (int(x_min), int(y_min - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
                        2)

        return image_display



