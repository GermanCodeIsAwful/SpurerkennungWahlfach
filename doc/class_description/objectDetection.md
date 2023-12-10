# Objekt Erkennung
## Funktionsweise

Die Objekterkennung wird in 2 Funktionen aufgeteilt. Das Bild wird nach der Kalibrierung zum Modell übergeben, um die Objekte zu erkennen. Dies wird an dieser Stelle gemacht, da die Objekterkennung nicht so gut funmktionuiert, wenn die erkannten Spuren bereits eingezeichnet wurden. Anders rum werden die Spuren auch nicht richtig erkannt, wenn die Linien der erkannten Objekte bereits eingezeichnet wurden.

Nachdem die Spuren eingezeichnet wurden, werden nun die im Vorhinein erkannten Objekte eingezeichnet.

## Code

In dieser Klasse werden Objekte auf einem Bild erkannt und bei Bedarf auch in das Bild eingezeichnet.

```python
import cv2
import torch
from ultralytics import YOLO
```

In diesem Abschnitt werden die erforderlichen Bibliotheken importiert. Das verwendete Neuronale Netzwerk wird
von Ultralytics zur Verfügung gestellt und mittels torch wird überprüft 
```python

class ObjectDetection:

    def __init__(self, debug):
        self.debug = debug
        self.model = YOLO(model='model/yolov8n.pt')
        
```
In diesem Abschnitt wird definiert welches Modell verwendet werden soll, und an welchem Ort es gespeichert ist bzw. gespeichert werden soll.

        
```python

    def get_objects(self, img):
        result = self.model(img)
        return result[0]
    
```

In dieser Funktion wird das Bild an das Modell übergeben. Da das Modell dafür ausgelegt ist, mehrere Bilder hintereindander zu analysieren, ist der Rückgabewert eine Liste. Daher gibt die Funktion nur das erste Element der Liste zurück. 
```python
0: 384x640 1 person, 3 cars, 1 truck, 1 traffic light, 122.0ms
Speed: 4.0ms preprocess, 122.0ms inference, 2.0ms postprocess per image at shape (1, 3, 384, 640)
ultralytics.engine.results.Boxes object with attributes:

cls: tensor([7., 2., 2., 2., 9., 0.])
conf: tensor([0.8744, 0.8730, 0.6189, 0.5645, 0.4814, 0.3888])
data: tensor([[2.7843e+02, 1.7056e+02, 4.0601e+02, 3.5747e+02, 8.7443e-01, 7.0000e+00],
        [1.7349e+02, 2.8269e+02, 2.4224e+02, 3.4159e+02, 8.7295e-01, 2.0000e+00],
        [4.9679e+01, 2.4959e+02, 1.6928e+02, 3.6506e+02, 6.1889e-01, 2.0000e+00],
        [2.2882e+02, 2.8456e+02, 2.7402e+02, 3.2901e+02, 5.6454e-01, 2.0000e+00],
        [2.7282e+02, 2.2767e+02, 2.8451e+02, 2.5972e+02, 4.8139e-01, 9.0000e+00],
        [5.5548e+02, 2.9634e+02, 5.6496e+02, 3.2160e+02, 3.8885e-01, 0.0000e+00]])
id: None
is_track: False
orig_shape: (374, 670)
shape: torch.Size([6, 6])
xywh: tensor([[342.2172, 264.0175, 127.5783, 186.9119],
        [207.8627, 312.1406,  68.7533,  58.8946],
        [109.4775, 307.3253, 119.5976, 115.4701],
        [251.4188, 306.7850,  45.2075,  44.4491],
        [278.6664, 243.6981,  11.6966,  32.0464],
        [560.2205, 308.9702,   9.4839,  25.2635]])
xywhn: tensor([[0.5108, 0.7059, 0.1904, 0.4998],
        [0.3102, 0.8346, 0.1026, 0.1575],
        [0.1634, 0.8217, 0.1785, 0.3087],
        [0.3753, 0.8203, 0.0675, 0.1188],
        [0.4159, 0.6516, 0.0175, 0.0857],
        [0.8361, 0.8261, 0.0142, 0.0675]])
xyxy: tensor([[278.4280, 170.5616, 406.0064, 357.4734],
        [173.4861, 282.6934, 242.2394, 341.5879],
        [ 49.6787, 249.5903, 169.2763, 365.0604],
        [228.8150, 284.5605, 274.0226, 329.0096],
        [272.8181, 227.6749, 284.5147, 259.7213],
        [555.4785, 296.3384, 564.9625, 321.6019]])
xyxyn: tensor([[0.4156, 0.4560, 0.6060, 0.9558],
        [0.2589, 0.7559, 0.3616, 0.9133],
        [0.0741, 0.6674, 0.2527, 0.9761],
        [0.3415, 0.7609, 0.4090, 0.8797],
        [0.4072, 0.6088, 0.4246, 0.6944],
        [0.8291, 0.7923, 0.8432, 0.8599]])

```

Hier ist ein Beispiel der Werte welche die Funktion getObjects(img) zurück gibt. Das Attribut CLS gibt an welche Klasse das erkannte Objekt hat. Das Attribut Conf gibt die Konfidenz an bezüglich des erkannten Objektes. Die letzten 4 Attribute sind 4 verschiedene Koordinatensysteme, wie die erkannten Objekte eingezeichnet werden können.

xywh: Dies sind die Koordinaten der Bounding Box im Format (x, y, Breite, Höhe).

xyxy: Dies sind die Koordinaten der Bounding Box im Format (x_min, y_min, x_max, y_max).

xyxyn: Dies sind die normalisierten Koordinaten der Bounding Box im Format (x_min_norm, y_min_norm, x_max_norm, y_max_norm), normalisiert auf den Bereich von 0 bis 1.

```python
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


```
In dieser Funktion wird das Bild und die Ergebnisse aus getObjects(img) übergeben.  
Als Erstes wird der Text gebildet, welcher an das erkannte Objekt angefügt wird. Hierzu wird die Nummer der Klasse dem Text zugewiesen

```python
label = f"{result.names[int(cls)]} {conf:.2f}"
```
Als Erstes wird der Text gebildet, welcher an das erkannte Objekt angefügt wird. Hierzu wird die Nummer der Klasse dem Text zugewiesen

```python
if conf <= 0.4:
    continue
```

Hier wird definiert, dass die Konfidenz mindestens 40% sein muss. Alle kleineren Werte haben in unseren Tests zu viele Fehlklassifikationen enthalten.

```python
cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
cv2.putText(img, label, (int(x_min), int(y_min - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),2)
```
In diesem Teil werden erst die erkannten Objekte eingezeichnet und anschließend der Text hinzugefügt
