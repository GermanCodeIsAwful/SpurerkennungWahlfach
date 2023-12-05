# PythonLaneLines_student.md

Dieses Programm wird von PythonServer_TCP_student.py aufgerufen, 
wenn eine Socket-Verbindung zur Simulation aufgebaut wurde. 
Im TCP-Modul wurde nur der Funktionsaufruf in diese Datei hinzugefügt.
Es dient als Konnektor zwischen slidingWindows.py und der oben genannten Verbindung.

## Code Analyse

Import der erforderlichen Pakete. 
Achtung: Import des SlidingWindow-Moduls. Dieser ist relativ organisiert, was nach dem aktuellen Standard 
nicht akzeptabel ist. Um dies zu beheben, sollte z.B. die Ordnerstruktur angepasst werden.

```python
import sys
import warnings

sys.path.append('../src')
from slidingWindows import SlidingWindows  # TODO: relative import NO GOOD IHHHHHHH PEP8 IS CRYING

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import json
```

Klassendefinition mit statischer Auflösung und Init-Funktion, in der allgemeine Parameter gesetzt werden.

```python
class Student:
    WIN_QHD = (640, 360)

    def __init__(self, debug):
        with open('../config/CarSim.json', 'r') as f:
            self.config = json.load(f)

        self.debug = debug
        self.slidingWindows = SlidingWindows(debug=debug, config=self.config["SLIDINGWINDOWS"],
                                             resolution=self.WIN_QHD)
```

Dieser Code Abschnitt ist ähnlich zu dem der features execute block von main.py.
Es wird jedoch keine Kalibrierung und keine FPS-Berechnung durchgeführt.
Dafür wird eine neue Funktion aufgerufen, die die notwendigen Parameter für den Simulator berechnet.

```python
    def find_lines(self, img):
        frame = cv.resize(img, self.WIN_QHD)
        left_points, right_points, mid_points = self.slidingWindows.start(frame)

        if self.debug:
            cv.circle(frame, self.config["SLIDINGWINDOWS"]["ROI_TL"], 3, (0, 0, 255), -1)
            cv.circle(frame, self.config["SLIDINGWINDOWS"]["ROI_TR"], 3, (0, 0, 255), -1)
            cv.circle(frame, self.config["SLIDINGWINDOWS"]["ROI_BL"], 3, (0, 0, 255), -1)
            cv.circle(frame, self.config["SLIDINGWINDOWS"]["ROI_BR"], 3, (0, 0, 255), -1)

        cv.polylines(frame, np.int32(left_points), isClosed=False, color=(255, 0, 0), thickness=2)
        cv.polylines(frame, np.int32(right_points), isClosed=False, color=(0, 255, 0), thickness=2)
        cv.polylines(frame, np.int32(mid_points), isClosed=False, color=(0, 0, 255), thickness=2)

        mean_coeff, radius, pos, distance = self._prep_return(mid_points)
        print(f'{mean_coeff} {radius} {pos}')
        if not radius:
            return frame, [0, 0, 0], 0, 0, 0

        cv.putText(frame, str(radius), (5, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4, cv.LINE_AA)
        cv.circle(frame, pos, 3, (255, 255, 255), -1)
        cv.circle(frame, (int(distance), self.WIN_QHD[1]), 5, (0, 0, 0), -1)

        return frame, mean_coeff, radius, pos, (distance - self.WIN_QHD[0])
```

Diese Funktion berechnet die folgenden Parameter für die Simulation:
- Krümmungsradius
- Position, an die das Auto fahren soll
- Abstand von der Mittellinie der Fahrspur

```python
    def _prep_return(self, points):
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                coeff = np.polyfit(points[0][:, 1], points[0][:, 0], 2)
                polynom = np.poly1d(coeff)
                drive_to_y = 200
                drive_to_x = polynom(drive_to_y)

                dy_dx = np.polyder(polynom)
                d2y_dx2 = np.polyder(dy_dx)
                curvature = (1 + (dy_dx(drive_to_y)) ** 2) ** 1.5 / abs(d2y_dx2(drive_to_y))

                distance_to_the_centre_line = polynom(self.WIN_QHD[1])
                return coeff, curvature, (int(drive_to_x), int(drive_to_y)), distance_to_the_centre_line
            except np.RankWarning:
                return False, False, False, False
```