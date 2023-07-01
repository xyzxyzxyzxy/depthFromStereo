import cv2 as cv

def undistort(img, K, dist): #returns roi after undistorting image
    h, w = img.shape[:2]
    newmat, ROI = cv.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    dst = cv.undistort(img, K, dist, None, newmat)
    x, y, w, h = ROI
    dst = dst[y:y+h, x:x+w]
    return dst, newmat