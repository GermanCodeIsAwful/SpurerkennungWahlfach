import cv2 as cv
import numpy as np
import os
import time
import json
from matplotlib import pyplot as plt


class slidingWindows():
    '''
    https://www.youtube.com/watch?v=ApYo6tXcjjQ
    https://github.com/mithi/advanced-lane-detection/blob/master/curves.py
    TODO: Krümmung der Fahrspur
    TODO: Abstand zur Mitte
    '''

    BOX_HEIGHT = 40
    MIN_PIXEL = 1
    def __init__(self, debug, config):
        self.debug = debug
        self.config = config

    def start(self, frame):

        h, w, _ = frame.shape
        inputPoints = np.float32([self.config["ROI_TL"], self.config["ROI_BL"], self.config["ROI_BR"], self.config["ROI_TR"]])
        outputPoints = np.float32([[0, 0], [0, h], [w, h], [w, 0]])

        perspectiveMatrix = cv.getPerspectiveTransform(inputPoints, outputPoints)
        frame = cv.warpPerspective(frame, perspectiveMatrix, (w, h))

        frame_gray_mask = cv.inRange(cv.cvtColor(frame[:, int(w*0.65):], cv.COLOR_BGR2GRAY), self.config["GRAY_MIN_THRESHOLD"], self.config["GRAY_MAX_THRESHOLD"])

        frame_hsv_mask = cv.inRange(cv.cvtColor(frame[:, :int(w*0.65)], cv.COLOR_BGR2HSV), tuple(self.config["HSV_MIN_THRESHOLD"]), tuple(self.config["HSV_MAX_THRESHOLD"]))

        frame_mask = np.concatenate((frame_hsv_mask, frame_gray_mask), axis=1)

        hist = np.sum(frame_mask[frame_mask.shape[0] // 2:, :], axis=0)
        mid = int(hist.shape[0] / 2)
        histLeftMax = np.argmax(hist[:mid])
        histRightMax = np.argmax(hist[mid:]) + mid

        # ---------------------------------------------

        boxStartY = h
        leftPositionsBox = []
        rightPositionsBox = []

        debug_mask = frame_mask.copy()
        nonzero = frame_mask.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        while boxStartY > 0:

            win_xleft_low = histLeftMax - self.config["WIDTH_ONESIDE"]
            win_xleft_high = histLeftMax + self.config["WIDTH_ONESIDE"]
            win_xright_low = histRightMax - self.config["WIDTH_ONESIDE"]
            win_xright_high = histRightMax + self.config["WIDTH_ONESIDE"]

            if self.debug:
                cv.rectangle(debug_mask, (int(win_xleft_low), boxStartY - self.BOX_HEIGHT), (int(win_xleft_high), boxStartY), (255, 255, 255), 2)
                cv.rectangle(debug_mask, (int(win_xright_low), boxStartY - self.BOX_HEIGHT), (int(win_xright_high), boxStartY), (255, 255, 255), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= boxStartY-self.BOX_HEIGHT) & (nonzeroy < boxStartY) & (nonzerox >= win_xleft_low) & (
                        nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= boxStartY-self.BOX_HEIGHT) & (nonzeroy < boxStartY) & (nonzerox >= win_xright_low) & (
                        nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            leftPositionsBox.append(good_left_inds)
            rightPositionsBox.append(good_right_inds)

            # If found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self.MIN_PIXEL:
                histLeftMax = np.mean(nonzerox[good_left_inds])
            if len(good_right_inds) > self.MIN_PIXEL:
                histRightMax = np.mean(nonzerox[good_right_inds])

            boxStartY -= self.BOX_HEIGHT

        leftPositionsBox = np.concatenate(leftPositionsBox)
        rightPositionsBox = np.concatenate(rightPositionsBox)

        leftx = nonzerox[leftPositionsBox]
        lefty = nonzeroy[leftPositionsBox]
        rightx = nonzerox[rightPositionsBox]
        righty = nonzeroy[rightPositionsBox]

        ploty = np.linspace(0, h - 1, h)
        invPerspectiveMatrix = cv.invert(perspectiveMatrix)[1]

        # TODO: POLYNOM ÜBERPRÜFEN

        if leftPositionsBox.any():
            left_fit = np.polyfit(lefty, leftx, 2)

            plotleftx = np.polyval(left_fit, ploty)

            left_points = np.array([np.transpose(np.vstack([plotleftx, ploty]))])

            left_points_original = cv.perspectiveTransform(left_points, invPerspectiveMatrix)
        else:
            left_points_original = np.array([[[0, 0]]])

        if rightPositionsBox.any():
            right_fit = np.polyfit(righty, rightx, 2)

            plotrightx = np.polyval(right_fit, ploty)

            right_points = np.array([np.transpose(np.vstack([plotrightx, ploty]))])

            right_points_original = cv.perspectiveTransform(right_points, invPerspectiveMatrix)
        else:
            right_points_original = np.array([[[0, 0]]])

        if self.debug:
            cv.imshow('Sliding Windows', debug_mask)

            if leftPositionsBox.any():
                left_points_birdseye = left_points.astype(np.int32)
                cv.polylines(frame, left_points_birdseye, isClosed=False, color=(255, 0, 0), thickness=8)

            if rightPositionsBox.any():
                right_points_birdseye = right_points.astype(np.int32)
                cv.polylines(frame, right_points_birdseye, isClosed=False, color=(0, 255, 0), thickness=8)

            cv.imshow('polyfit', frame)


        return left_points_original.astype(np.int32), right_points_original.astype(np.int32)