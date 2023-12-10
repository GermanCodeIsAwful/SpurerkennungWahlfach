import cv2 as cv
import numpy as np
import time
import json
import math

from src import cameraCalibration, slidingWindows,objectDetection_Y8


class Main:
    WIN_QHD = (640, 360)  # h mehrfaches von 40

    def __init__(self, config, videoPath, debug=False, objectDetection=False):

        print('starting init ...')

        with open(config, 'r') as f:
            self.config = json.load(f)
        self.debug = debug
        self.videoPath = videoPath
        self.activateobjectdetection = objectDetection

        # Define Features

        self.calibration = cameraCalibration.CameraCalibration(path_calib='img/Udacity/calib', inner_row=9,
                                                               inner_coll=6, debug=debug, resolution=self.WIN_QHD)
        self.slidingWindows = slidingWindows.SlidingWindows(debug=debug, config=self.config["SLIDINGWINDOWS"],
                                                            resolution=self.WIN_QHD)
        if self.activateobjectdetection:
            self.objectDetection = objectDetection_Y8.ObjectDetection(True)
        self.fps_counter = 0
        self.fps = 'wait'  # für ersten ~6 Frames sekunden

        print('done init')

    def loadMp4(self):

        print('mit "q" das Video abbrechen')

        video = cv.VideoCapture(self.videoPath)
        prevFrameTime = time.time()
        font = cv.FONT_HERSHEY_SIMPLEX

        while (video.isOpened()):
            frameAvailable, frame = video.read()

            if not frameAvailable:
                break

            # ------------ execute features here ------------

            frame = cv.resize(frame, self.WIN_QHD)

            if self.debug:
                cv.imshow('raw', frame)

            frame = self.calibration.get_calib_img(frame)
            if self.activateobjectdetection:
                results = self.objectDetection.get_objects(frame)

            left_points_original, right_points_original, _ = self.slidingWindows.start(frame)

            left_points_original, right_points_original = left_points_original.astype(
                np.int32), right_points_original.astype(np.int32)

            if self.debug:
                cv.circle(frame, self.config["SLIDINGWINDOWS"]["ROI_TL"], 3, (0, 0, 255), -1)
                cv.circle(frame, self.config["SLIDINGWINDOWS"]["ROI_TR"], 3, (0, 0, 255), -1)
                cv.circle(frame, self.config["SLIDINGWINDOWS"]["ROI_BL"], 3, (0, 0, 255), -1)
                cv.circle(frame, self.config["SLIDINGWINDOWS"]["ROI_BR"], 3, (0, 0, 255), -1)

            # ------------ done -> fps ------------

            if self._calcFps(time.time(), prevFrameTime):
                prevFrameTime = time.time()

            cv.putText(frame, self.fps, (5, 50), font, 2, (255, 255, 255), 4, cv.LINE_AA)

            cv.polylines(frame, left_points_original, isClosed=False, color=(255, 0, 0), thickness=2)
            cv.polylines(frame, right_points_original, isClosed=False, color=(0, 255, 0), thickness=2)

            if self.activateobjectdetection:
                frame = self.objectDetection.draw_boxes_cv2(frame, results)

            cv.imshow('Video', frame)

            # https://learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/
            if cv.waitKey(25) & 0xFF == ord('q'):
                break

        video.release()
        cv.destroyAllWindows()

    def _calcFps(self, newFrameTime, prevFrameTime):
        '''
        https://www.geeksforgeeks.org/python-displaying-real-time-fps-at-which-webcam-video-file-is-processed-using-opencv/
        '''

        self.fps_counter += 1

        if (newFrameTime - prevFrameTime) > 0.25:  # avg. über 0.25 sec
            fps = math.ceil(self.fps_counter / (newFrameTime - prevFrameTime))
            self.fps_counter = 0
            fps = str(fps)

            self.fps = fps

            return True

        else:
            return False


if __name__ == '__main__':
    mode = input('Welches Video soll abgespielt werden (1,2,3)?:')

    if mode == '1':
        main = Main("config/video.json", "img/Udacity/project_video.mp4", False, True)
        main.loadMp4()

    elif mode == '2':
        main_harder = Main("config/video_harder.json", "img/Udacity/challenge_video.mp4", True, False)
        main_harder.loadMp4()

    elif mode == '3':
        main_harder_c = Main("config/video_harder_challenge.json",
                             "img/Udacity/harder_challenge_video.mp4", True, False)
        main_harder_c.loadMp4()

    else:
        print('Bitte wählen Sie "1", "2" oder "3"')
