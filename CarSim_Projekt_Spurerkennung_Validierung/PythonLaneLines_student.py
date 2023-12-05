# Project Lane Detection

# necessary imports
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


# parameters:
# img: image to apply lane detection on

class Student:
    WIN_QHD = (640, 360)

    def __init__(self, debug):
        with open('../config/CarSim.json', 'r') as f:
            self.config = json.load(f)

        self.debug = debug
        self.slidingWindows = SlidingWindows(debug=debug, config=self.config["SLIDINGWINDOWS"],
                                             resolution=self.WIN_QHD)

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
