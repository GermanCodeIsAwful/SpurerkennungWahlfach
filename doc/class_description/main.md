# Main Class

Beispiel:

```python 
class Main():
    WIN_QHD = (640, 360) 

    def __init__(self, config, videoPath, debug=False):

        with open(config, 'r') as f:
            self.config = json.load(f)
        self.debug = debug
        self.videoPath = videoPath
        self.calibration = cameraCalibration.cameraCalibration(debug=debug)
        self.transformation = transformation.transformation(debug=debug)
        self.slidingWindows = slidingWindows.slidingWindows(debug=debug, 
                                                            config=self.config["SLIDINGWINDOWS"], 
                                                            resolution=self.WIN_QHD)
        self.output = programmeOutput.output(debug=debug)

        self.fps_counter = 0
        self.fps = 'wait' 
```