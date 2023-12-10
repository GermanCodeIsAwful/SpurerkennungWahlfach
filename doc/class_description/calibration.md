# Kalibrierung

## Funktionsweise

## Code

```python 
class cameraCalibration():
    '''
    TODO:   Entzerrung des input
    '''

    def __init__(self, path_calib, inner_row, inner_coll, debug, resolution):
        self.debug = debug
        self._get_camera_calibration(path_calib, inner_row, inner_coll)
        self.resolution = resolution

    def _show(self, img):
        plt.close()
        plt.figure()
        plt.imshow(img)
        plt.show()
        
    def _get_camera_calibration(self, path_calib, inner_row, inner_coll):
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objp = np.zeros((inner_coll * inner_row, 3), np.float32)
        objp[:, :2] = np.mgrid[0:inner_row, 0:inner_coll].T.reshape(-1, 2)

        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        for imgn in os.listdir(path_calib):
            img = cv.imread(path_calib + "/" + str(imgn))
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            retval, corners = cv.findChessboardCorners(image=img_gray, patternSize=(inner_row, inner_coll), flags=None)
            if retval:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                cv.drawChessboardCorners(img, (inner_row, inner_coll), corners2, retval)

        ret, mtx, dist, _, _ = cv.calibrateCamera(objpoints, imgpoints, img_gray.shape[::-1], None, None)

        if self.debug:
            print(f'ret: {ret} mtx: {mtx} dist: {dist}')

        self.mtx = mtx
        self.dist = dist

    def get_calib_img(self, img):
        h, w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))
        dst = cv.undistort(img, self.mtx, self.dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        frame = cv.resize(dst[y:y + h, x:x + w], self.resolution)
        return frame

##Anwendung:
# Initialisierung: ret, mtx, dist, rvecs, tvecs = get_camera_calibration(path_calib,inner_row,inner_coll)
# Pro Frame new_img = get_calib_img(img,mtx,dist)


```