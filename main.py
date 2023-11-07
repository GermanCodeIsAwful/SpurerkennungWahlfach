import cv2 as cv
import numpy as np
import os
from src import cameraCalibration, transformation, slidingWindows, programmeOutput

class Main():
    '''
    TODO: Umwandlung von Bildern und Videos in verarbeitbares Material
    '''
    def __init__(self, debug=False):
        print('Willkommen beim Projekt "Erkennung von Spurmarkierungen"')

        # Define Features

        self.calibration = cameraCalibration.cameraCalibration(debug=debug)
        self.transformation = transformation.transformation(debug=debug)
        self.slidingWindows = slidingWindows.slidingWindows(debug=debug)
        self.output = programmeOutput.output(debug=debug)

    def dosth (self):
        return 'Test'

if __name__ == '__main__':
    main = Main()

    main.dosth()
