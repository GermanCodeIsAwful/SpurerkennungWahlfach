import cv2 as cv
import numpy as np
import warnings


class SlidingWindows:
    '''
    https://www.youtube.com/watch?v=ApYo6tXcjjQ
    https://github.com/mithi/advanced-lane-detection/blob/master/curves.py
    TODO: Krümmung der Fahrspur
    TODO: Abstand zur Mitte
    TODO: POLYNOM ÜBERPRÜFEN
    '''

    BOX_HEIGHT = 40
    MIN_PIXEL = 1

    def __init__(self, debug, config, resolution):
        self.debug = debug
        self.config = config
        self.w = resolution[0]
        self.h = resolution[1]
        self.perspectiveMatrix, self.invPerspectiveMatrix = self._gen_matrix()
        self.ploty = np.linspace(0, self.h, int(self.h / 4))
        self.prev_left_lane_coeff = []
        self.prev_right_lane_coeff = []
        self.split_value = 0.65 if self.config["SPLIT_LANE"] else 0.5

        self.test = []

    def start(self, frame):

        # ROI

        frame = cv.warpPerspective(frame, self.perspectiveMatrix, (self.w, self.h))

        # threshold

        frame_gray_mask = cv.inRange(cv.cvtColor(frame[:, int(self.w * self.split_value):], cv.COLOR_BGR2GRAY),
                                     self.config["GRAY_MIN_THRESHOLD"], self.config["GRAY_MAX_THRESHOLD"])

        frame_hsv_mask = cv.inRange(cv.cvtColor(frame[:, :int(self.w * self.split_value)], cv.COLOR_BGR2HSV),
                                    tuple(self.config["HSV_MIN_THRESHOLD"]), tuple(self.config["HSV_MAX_THRESHOLD"]))

        frame_mask = np.concatenate((frame_hsv_mask, frame_gray_mask), axis=1)

        # histogram

        hist = np.sum(frame_mask[frame_mask.shape[0] // 2:, :], axis=0)
        mid = int(hist.shape[0] / 2)
        histLeftMax = np.argmax(hist[:mid])
        histRightMax = np.argmax(hist[mid:]) + mid

        # Start Sliding Windows

        boxStartY = self.h
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
                cv.rectangle(debug_mask, (int(win_xleft_low), boxStartY - self.BOX_HEIGHT),
                             (int(win_xleft_high), boxStartY), (255, 255, 255), 2)
                cv.rectangle(debug_mask, (int(win_xright_low), boxStartY - self.BOX_HEIGHT),
                             (int(win_xright_high), boxStartY), (255, 255, 255), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = \
                ((nonzeroy >= boxStartY - self.BOX_HEIGHT) & (nonzeroy < boxStartY) & (nonzerox >= win_xleft_low) & (
                        nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = \
                ((nonzeroy >= boxStartY - self.BOX_HEIGHT) & (nonzeroy < boxStartY) & (nonzerox >= win_xright_low) & (
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

        # back transformation from roi and polyfit

        if leftPositionsBox.any():
            left_points_original = self._rate_polynom(nonzerox[leftPositionsBox], nonzeroy[leftPositionsBox], True)

        else:
            left_points_original = np.array([[[0, 0]]])

        if rightPositionsBox.any():
            right_points_original = self._rate_polynom(nonzerox[rightPositionsBox], nonzeroy[rightPositionsBox], False)

        else:
            right_points_original = np.array([[[0, 0]]])

        # show debug

        if self.debug:
            cv.imshow('Sliding Windows', debug_mask)

            if leftPositionsBox.any():
                left_points_birdseye = cv.perspectiveTransform(left_points_original, self.perspectiveMatrix).astype(
                    np.int32)
                cv.polylines(frame, left_points_birdseye, isClosed=False, color=(255, 0, 0), thickness=8)

            if rightPositionsBox.any():
                right_points_birdseye = cv.perspectiveTransform(right_points_original, self.perspectiveMatrix).astype(
                    np.int32)
                cv.polylines(frame, right_points_birdseye, isClosed=False, color=(0, 255, 0), thickness=8)

            cv.imshow('polyfit', frame)

        # return
        return (left_points_original, right_points_original,
                self._transform_points(self.prev_left_lane_coeff * 0.5 + self.prev_right_lane_coeff * 0.5))

    def _gen_matrix(self):
        inputPoints = np.float32(
            [self.config["ROI_TL"], self.config["ROI_BL"], self.config["ROI_BR"], self.config["ROI_TR"]])
        outputPoints = np.float32([[0, 0], [0, self.h], [self.w, self.h], [self.w, 0]])

        perspectiveMatrix = cv.getPerspectiveTransform(inputPoints, outputPoints)
        invPerspectiveMatrix = cv.invert(perspectiveMatrix)[1]

        return perspectiveMatrix, invPerspectiveMatrix

    def _rate_polynom(self, x, y, leftorright: bool):
        '''
        leftorright: left-True; right-False
        '''

        prev_lane_coeff = self.prev_left_lane_coeff if leftorright else self.prev_right_lane_coeff

        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                coefficients = np.polyfit(y, x, 2)
            except np.RankWarning:
                coefficients = prev_lane_coeff

        if any(prev_lane_coeff):
            if not (self.config["COEFF_THRESHOLD"] > coefficients[0] > -self.config["COEFF_THRESHOLD"]):
                # coeff 1 : +1 -1

                coefficients = prev_lane_coeff
            else:
                coefficients[0] = coefficients[0] * 0.9 + prev_lane_coeff[0] * 0.1

        setattr(self, 'prev_left_lane_coeff' if leftorright else 'prev_right_lane_coeff', coefficients)

        return self._transform_points(coefficients)

    def _transform_points(self, coefficients):
        plotx = np.polyval(coefficients, self.ploty)
        points = np.array([np.transpose(np.vstack([plotx, self.ploty]))])
        points_original = cv.perspectiveTransform(points, self.invPerspectiveMatrix)

        return points_original
