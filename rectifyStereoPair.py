import numpy as np
import glob
import cv2 as cv
import undistort
import matplotlib.pyplot as plt
import sys
import getopt
from epipolarGeometry import *

INDEX = None
SHOW_EPILINES = 0
CORRECT_LENS_DISTORTION = 0
CHECK_EPIPOLAR_CONSTRAINT = 0
DRAW_MATCHES = 0
SHOW_ROIS = 0
BRUTE_FORCE_MATCHING = 0
RX = None
USE_KITTI = 0
nMatches = 100

try:
    optlist, args = getopt.getopt(sys.argv[1:], 'R:', ['index=', 'show-epi', 'undistort', 
                                                        'check-epi', 
                                                        'show-matches',
                                                        'show-rois',
                                                        'use-kitti',
                                                        'bruteforce='])
except:
    print('unexpected argument')
    sys.exit()

print(optlist)
print(args)

for opt, arg in optlist:
    if opt in ['--use-kitti']:
        USE_KITTI = 1
    elif opt in ['--index']:
        INDEX = int(arg)
    elif opt in ['--show-epi']:
        SHOW_EPILINES = 1
    elif opt in ['--undistort']:
        CORRECT_LENS_DISTORTION = 1
    elif opt in ['--check-epi']:
        CHECK_EPIPOLAR_CONSTRAINT = 1
    elif opt in ['--show-matches']:
        DRAW_MATCHES = 1
    elif opt in ['--show-rois']:
        SHOW_ROIS = 1
    elif opt in ['--bruteforce']:
        BRUTE_FORCE_MATCHING = 1
        nMatches = int(arg)
    elif opt in ['-R']:
        RX = int(arg)

if INDEX == None:
    print("Stereo pair index is required")
    sys.exit()

#load stereo pair with index INDEX
if USE_KITTI:
    filename = "./KITTIPairs/*{:03d}.png".format(INDEX) #if using kitti intrinsic parameters load pair from KITTI dir
    data = np.load("intrinsicParametersKITTI.npz")
    K1 = data['K1']
    K2 = data['K2']
    dist1 = data['dist1']
    dist2 = data['dist2']
    w = int(data['w'])
    h = int(data['h'])
else:
    filename = "./StereoPairs/*{:02d}.JPG".format(INDEX)
    data = np.load("intrinsicParameters.npz")
    K1 = data['K']
    K2 = K1
    dist1 = data['dist']
    dist2 = dist1
    w = int(data['w'])
    h = int(data['h'])

K = np.array([K1, K2])
dist = np.array([dist1, dist2])

print(filename)
stereo_pair = glob.glob(filename)
print(f"Loading stereo pair: {stereo_pair}")
print(f"Loading intrinsic parameters")

print("\nIntrinsic parameter matrix camera 1: \n", K1)
print("\nIntrinsic parameter matrix camera 2: \n", K2)
print("\nDistortion coefficients camera 1: \n", dist1)
print("\nDistortion coefficients camera 2: \n", dist2)
print("\nImage_size: \n", w, h)

stereo = []
stereo_color = []

for i, fname in enumerate(stereo_pair):
    img_color = cv.imread(fname)
    img_color = cv.cvtColor(img_color, cv.COLOR_BGR2RGB)
    img_color = cv.resize(img_color, (w, h), interpolation = cv.INTER_AREA)
    img_gray = cv.cvtColor(img_color, cv.COLOR_RGB2GRAY)
    if CORRECT_LENS_DISTORTION:
        print("Undistorting stereo pair, refining intrinsic parameters...\n")
        img_gray, newmat = undistort.undistort(img_gray, K[i], dist[i])
        img_color, _ = undistort.undistort(img_color, K[i], dist[i])
        print(f"Shape after undistortion using intrinsic parameters: {img_gray.shape}")
        print(f"Shape after undistortion using intrinsic parameters (color): {img_color.shape}")
        K[i] = newmat
        print("\nNew Intrinsic parameter matrix: \n", K[i])
    stereo.append(img_gray)
    stereo_color.append(img_color)
    cv.imshow(fname, img_gray)
    cv.waitKey(0)

imright = stereo[0]
imleft = stereo[1]
imright_color = stereo_color[0]
imleft_color = stereo_color[1]

h, w = imright.shape #new height and with of both images

print(f"imright type: {imright.dtype}")

pts1, pts2 = findCorrespondence(imleft, imright, BRUTE_FORCE_MATCHING, nMatches, DRAW_MATCHES)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

#F, mask = cv.findFundamentalMat(pts1 ,pts2, cv.FM_RANSAC, 3, 0.99) 
F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)

pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

print("\nFundamental matrix: \n", F)
#print("points1: \n", pts1.reshape(-1, 1, 2), "\npoints2: \n", pts2.reshape(-1, 1, 2))
#print(len(pts1), len(pts2))

if SHOW_EPILINES:
    #compute epilines
    l1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    l1 = l1.reshape(-1, 3)
    epi_left, epi_leftc = drawlines(imleft, imright, l1, pts1, pts2)


    l2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    l2 = l2.reshape(-1, 3)
    epi_right, epi_rightc = drawlines(imright, imleft, l2, pts2, pts1)

    fig, ax = plt.subplots(1, 2, figsize=(12,7))
    ax[0].imshow(epi_left)
    ax[1].imshow(epi_right)
    plt.show()

    # print(f"epilines on left image: {l1}")
    # print(f"epilines on right image: {l2}")
    # print(f"lines1 lenght: {l1.shape}, #of points right image: {pts2.shape}")
    # print(f"lines2 lenght: {l2.shape}, #of points left image: {pts1.shape}")

#CHECK QUALITY USING EPIPOLAR CONSTRAINT

if CHECK_EPIPOLAR_CONSTRAINT:
    checkEpipolarConst(pts1, pts2, l1, l2, F)

#get essential matrix from fundamental matrix
E = K[0].T @ F @ K[0] 

E1 = K[1].T @ F @ K[1] # if K1 and K2 are the same E and E1 are the same

print(f"\nEssential matrix: \n{E}")

R1, R2, t, = cv.decomposeEssentialMat(E1)

#DEBUG
# t[1:3] = 0
# t[0] = -4.731049999999999978e-01
# t[1] = 5.551470000000000189e-03
# t[2] = -5.250882000000000119e-03
# R1 = np.zeros((3, 3))
# theta = np.deg2rad(5)
# R1[0, 0] = np.cos(theta)
# R1[0, 2] = -np.sin(theta)
# R1[1, 1] = 1
# R1[2, 0] = np.sin(theta)
# R1[2, 2] = np.cos(theta)
# print("DEBUG T: ", t)
# print("DEBUG R1: ", R1)

print(f"\nR1: {R1}, \nR2: {R2}, \nt: {t}")

if RX == 1:
    R = R1
elif RX == 2:
    R = R2
else:
    print('argument -r has to be either 1 or 2')
    print('quitting')
    sys.exit()

#STEREORECTIFY
rectLeft, rectRight, projectionLeft, projectionRight, Q, roiL, roiR = cv.stereoRectify(K[0], dist[0], K[1], dist[1], 
                                                                                       (w, h), R, t, 
                                                                                       None, None, None, None, None, 
                                                                                       cv.CALIB_ZERO_DISPARITY, 1, (0, 0))

print(f"\nrectificationTransL: \n{rectLeft}\n")
print(f"rectificationTransR: \n{rectRight}\n")
print(f"projectionMatrixL: \n{projectionLeft}\n")
print(f"projectionMatrixR: \n{projectionRight}\n")
print(f"Q: \n{Q}")

xl, yl, wl, hl = roiL
xr, yr, wr, hr = roiR

print("roiL values: ", roiL)
print("roiR values: ", roiR)

mapxL, mapyL = cv.initUndistortRectifyMap(K[0], dist[0], rectLeft, projectionLeft, (w, h), cv.CV_32FC1)
mapxR, mapyR = cv.initUndistortRectifyMap(K[1], dist[1], rectRight, projectionRight, (w, h), cv.CV_32FC1)

#used to get color in the point cloud
rectifiedLeft_color = cv.remap(imleft_color, mapxL, mapyL, cv.INTER_LANCZOS4)
rectifiedRight_color = cv.remap(imright_color, mapxR, mapyR, cv.INTER_LANCZOS4)

rectifiedLeft = cv.remap(imleft, mapxL, mapyL, cv.INTER_LANCZOS4)
rectifiedRight = cv.remap(imright, mapxR, mapyR, cv.INTER_LANCZOS4)

if np.any(roiL) and np.any(roiR):
    rectLeftRoi = rectifiedLeft[xl:xl+hl, yl:yl + wl]
    rectRightRoi = rectifiedRight[xr:xr+hr, yr:yr + wr]
    rectLeftRoi_color = rectifiedLeft_color[xl:xl+hl, yl:yl + wl]
    rectRightRoi_color = rectifiedRight_color[xr:xr+hr, yr:yr + wr]
else:
    rectLeftRoi = rectifiedLeft
    rectRightRoi = rectifiedRight
    rectLeftRoi_color = rectifiedLeft_color
    rectRightRoi_color = rectifiedRight_color

#show grayscale rectified images
cv.imshow('rectL', rectifiedLeft)
cv.waitKey(0)
cv.imshow('rectR', rectifiedRight)
cv.waitKey(0)

if SHOW_ROIS and (np.any(roiL) and np.any(roiR)):
    cv.imshow('roiL', rectLeftRoi)
    cv.waitKey(0)
    cv.imshow('roiR', rectRightRoi)
    cv.waitKey(0)

cv.destroyAllWindows()

fname = "stereoData" + str(INDEX) + ".npz"
cv.imwrite('rectifiedLeft.jpg', rectifiedLeft)
cv.imwrite('rectifiedRight.jpg', rectifiedRight)
cv.imwrite('rectifiedLeft_c.jpg', rectifiedLeft_color)
cv.imwrite('rectifiedRight_c.jpg', rectifiedRight_color)
np.savez(fname, F=F, E=E, R1=R1, t=t, Q=Q, roiL=roiL, roiR=roiR)