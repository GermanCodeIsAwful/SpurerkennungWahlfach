import cv2 as cv
import numpy as np
import os
import time
import json

from src import cameraCalibration, transformation, slidingWindows, programmeOutput

class Main():

    WIN_QHD = (640, 360)    # h mehrfaches von 40

    def __init__(self, config, videoPath, debug=False,):

        with open(config, 'r') as f:
            self.config = json.load(f)
        self.debug = debug
        self.videoPath = videoPath

        # Define Features

        self.calibration = cameraCalibration.cameraCalibration(debug=debug)
        self.transformation = transformation.transformation(debug=debug)
        self.slidingWindows = slidingWindows.slidingWindows(debug=debug, config=self.config["SLIDINGWINDOWS"])
        self.output = programmeOutput.output(debug=debug)



    def loadMp4(self):

        video = cv.VideoCapture(self.videoPath)
        prevFrameTime = 0
        newFrameTime = 0
        font = cv.FONT_HERSHEY_SIMPLEX

        while (video.isOpened()):
            frameAvailable, frame = video.read()

            if not frameAvailable:
                break

            # ------------ execute features here ------------

            frame = cv.resize(frame, self.WIN_QHD)
            left_points_original, right_points_original = self.slidingWindows.start(frame)

            if self.debug:
                cv.circle(frame, self.config["SLIDINGWINDOWS"]["ROI_TL"], 3, (0, 0, 255), -1)
                cv.circle(frame, self.config["SLIDINGWINDOWS"]["ROI_TR"], 3, (0, 0, 255), -1)
                cv.circle(frame, self.config["SLIDINGWINDOWS"]["ROI_BL"], 3, (0, 0, 255), -1)
                cv.circle(frame, self.config["SLIDINGWINDOWS"]["ROI_BR"], 3, (0, 0, 255), -1)

            # ------------ done -> fps ------------

            newFrameTime = time.time()
            fps = self._calcFps(newFrameTime, prevFrameTime)
            prevFrameTime = newFrameTime

            cv.putText(frame, fps, (5, 50), font, 2, (255, 255, 255), 4, cv.LINE_AA)

            cv.polylines(frame, left_points_original, isClosed=False, color=(255, 0, 0), thickness=2)
            cv.polylines(frame, right_points_original, isClosed=False, color=(0, 255, 0), thickness=2)

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

        fps = 1 / (newFrameTime - prevFrameTime)
        fps = str(int(fps))

        return fps

if __name__ == '__main__':

    main = Main("config/video.json", "img/Udacity/project_video.mp4", True)
    main_harder = Main("config/video_harder.json", "img/Udacity/challenge_video.mp4", True)
    main_harder_c = Main("config/video_harder_challenge.json", "img/Udacity/harder_challenge_video.mp4", True)

    main.loadMp4()
    #main_harder.loadMp4()
    #main_harder_c.loadMp4()