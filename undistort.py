import cv2 as cv
import numpy as np
import glob
import sys

def undistort(img, K, dist): #returns roi after undistorting image
    h, w = img.shape[:2]
    newmat, ROI = cv.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    dst = cv.undistort(img, K, dist, None, newmat)
    x, y, w, h = ROI
    dst = dst[y:y+h, x:x+w]
    return dst, newmat

def main():
    if len(sys.argv) < 2:
        print("Calibration images directory name required")
        sys.exit()

    DIR = sys.argv[1]
    imgs = glob.glob(DIR+"/*jpg")
    data = np.load("intrinsicParameters.npz")
    K = data['K']
    dist = data['dist']
    w = int(data['w'])
    h = int(data['h'])

    for fname in imgs:
        img = cv.imread(fname)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, (w, h))
        undistorted, _ = undistort(img, K, dist)
        if undistorted.shape[0] > 0 and undistorted.shape[1] > 0:
            cv.imshow('undistorted', undistorted)
            cv.waitKey(0)

if __name__ == '__main__':
    main()