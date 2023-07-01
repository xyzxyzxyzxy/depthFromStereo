import numpy as np
import cv2 as cv
import glob

square_sz = 25 #25mm
p_size = (7, 5)

#termination criteria
crit = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((p_size[0]*p_size[1], 3), np.float32) #points in world coordinates [X, Y, Z]
objp[:, :2] = np.mgrid[0:p_size[0], 0:p_size[1]].T.reshape(-1, 2)*square_sz

objectpoints = [] #stores vectors of points for ALL the calibration images
imagepoints = []

imgs = glob.glob("./CalibrationImages1/*jpg") #directory with images of the calibration pattern

goodImgcount = 0

for fname in imgs:
    #read and convert to grayscale
    img = cv.imread(fname)
    #resize
    img = cv.resize(img,(0, 0),fx=0.12, fy=0.12, interpolation = cv.INTER_AREA)
    print(img.shape)

    #hsv mask
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lwr = np.array([0, 0, 143])
    upr = np.array([179, 61, 252])
    msk = cv.inRange(img_hsv, lwr, upr)

    #use morphology
    krn = cv.getStructuringElement(cv.MORPH_RECT, (50, 50))
    dlt = cv.dilate(msk, krn, iterations=15)
    pattern = 255 - cv.bitwise_and(dlt, msk)
    # cv.imshow("mask", pattern)
    # cv.waitKey(0)

    print(f"searching for corners in image: {fname}")

    #find corners
    ret, corners = cv.findChessboardCorners(np.uint8(pattern), p_size, None)

    print(f"corners found: {ret}")
    
    if ret == True:
        objectpoints.append(objp) #these [X, Y, Z] are the same for all images! Pattern does not move
        ref_corners = cv.cornerSubPix(pattern, corners, (11, 11), (-1, -1), crit)
        imagepoints.append(ref_corners)
        #ret is passed from the find corners function to understand if the whole board was found or not
        cv.drawChessboardCorners(img, p_size, ref_corners, ret)
        cv.imshow('corners', img)
        cv.waitKey(30)
        goodImgcount = goodImgcount + 1

    cv.destroyAllWindows()

#now the calibration
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, mat, dist, rvecs, tvecs = cv.calibrateCamera(objectpoints, imagepoints, gray.shape[::-1], None, None)

#compute reprojection error
mean_error = 0
for i in range(len(objectpoints)):
        #projection of the given points using found intrinsic/extrinsic param
    imagepoints2, _ = cv.projectPoints(objectpoints[i], rvecs[i], tvecs[i], mat, dist)
    error = cv.norm(imagepoints[i], imagepoints2, cv.NORM_L2)/len(imagepoints2)
    mean_error = mean_error + error

print(f"\ntotal reprojection error: {mean_error}\n")
print(f"Intrinsic parameters: \n{mat}")
print(f"Distortion coeff: \n{dist}")

#size of sensor
sensor = (0.0223, 0.0149) #in meters

w = img.shape[1]
h = img.shape[0]

print("horizontalpx: ", w)
print("verticalpx: ", h)
print("total # of px: ", w*h)

pxsize = sensor[0]/w
pysize = sensor[1]/h

#print(f"size of pixel px: {pxsize}, size of pixel py: {pysize}")

# fx = mat[0, 0]*pxsize
# fy = mat[1, 1]*pysize

# print(f"estimated focal length fx: {fx}, fy: {fy}")

print("# of good images: ", goodImgcount)

np.savez("intrinsicParameters.npz", K=mat, dist=dist, w=w, h=h)
