import numpy as np
import glob
import cv2 as cv
import undistort
import matplotlib.pyplot as plt
import sys
import getopt
from epipolarGeometry import *

(FLANN_MATCHING, BRUTEFORCE_MATCHING, MANUAL_MATCHING) = range(3)
# DEFAULT OPTIONS
INDEX = None
SHOW_EPILINES = 0
FIND_FUNDAMENTAL = 0
CORRECT_LENS_DISTORTION = 0
DRAW_MATCHES = 0
SHOW_ROIS = 0
MATCHING = FLANN_MATCHING
RX = 1
USE_KITTI = 0
FORCE_POSE = 0

nMatches = 100 #Used only when matching is bruteforce

try:
    optlist, args = getopt.getopt(sys.argv[1:], '', ['index=', 'show-epi', 'undistort', 
                                                        'find-fundamental', 
                                                        'show-matches',
                                                        'show-rois',
                                                        'use-kitti',
                                                        'force-pose',
                                                        'manual-matching',
                                                        'bruteforce='])
except:
    print('unexpected argument')
    sys.exit()

print(optlist)
print(args)

for opt, arg in optlist:
    #use sample from kitti dataset with kitti calibration data
    if opt in ['--use-kitti']:
        USE_KITTI = 1
    elif opt in ['--index']:
        INDEX = int(arg)
    elif opt in ['--show-epi']:
        SHOW_EPILINES = 1
    #perform undistortion using intrinsic paramters / distortion coefficients
    elif opt in ['--undistort']:
        CORRECT_LENS_DISTORTION = 1
    #find fundamental matrix instead of essential matrix directly
    elif opt in ['--find-fundamental']:
        FIND_FUNDAMENTAL = 1
    #displays found matches over images
    elif opt in ['--show-matches']:
        DRAW_MATCHES = 1
    elif opt in ['--show-rois']:
        SHOW_ROIS = 1
    elif opt in ['--bruteforce']:
        MATCHING = BRUTEFORCE_MATCHING
        nMatches = int(arg)
    elif opt in ['--manual-matching']:
        MATCHING = MANUAL_MATCHING
    elif opt in ['--force-pose']:
        FORCE_POSE = 1

if INDEX == None:
    print("Stereo pair index is required")
    sys.exit()

#load stereo pair with index INDEX
if USE_KITTI:
    filename = "./KITTIPairs/*{:03d}.png".format(INDEX) #if using kitti intrinsic parameters load pair from KITTI dir
    print("Loading KITTI intrinsic parameters...")
    data = np.load("intrinsicParametersKITTI.npz")
    K1 = data['K1']
    K2 = data['K2']
    dist1 = data['dist1']
    dist2 = data['dist2']
    w_calib = int(data['w'])
    h_calib = int(data['h'])
else:
    filename = "./StereoPairs/*{:02d}.JPG".format(INDEX)
    print(f"Loading intrinsic parameters...")
    data = np.load("intrinsicParameters.npz")
    K1 = data['K']
    K2 = K1
    dist1 = data['dist']
    dist2 = dist1
    w_calib = int(data['w'])
    h_calib = int(data['h'])

K = np.array([K1, K2])
dist = np.array([dist1, dist2])

stereo_pair = glob.glob(filename)
#force left right order
stereo_pair = sorted(stereo_pair)
print(f"Loading stereo pair: {stereo_pair}")
print("Camera Intrinsic Parameters:\n")
print("Intrinsic parameter matrix camera 1: \n", K1)
print("Intrinsic parameter matrix camera 2: \n", K2)
print("Distortion coefficients camera 1: \n", dist1)
print("Distortion coefficients camera 2: \n", dist2)
print("Image_size: \n", w_calib, h_calib)

stereo = []
stereo_color = []

for i, fname in enumerate(stereo_pair):
    img = cv.imread(fname)
    img = cv.resize(img, (w_calib, h_calib), interpolation = cv.INTER_AREA)
    
    if CORRECT_LENS_DISTORTION:
        print("\nUndistorting stereo pair, updating intrinsic parameters...\n")
        img, newmat = undistort.undistort(img, K[i], dist[i])
        print(f"Shape after undistortion using intrinsic parameters: {img.shape}")
        K[i] = newmat
        print("\nNew Intrinsic parameter matrix: \n", K[i])
    
    stereo.append(cv.cvtColor(img, cv.COLOR_RGB2GRAY))
    stereo_color.append(img)
    cv.imshow(fname, img)
    cv.waitKey(0)

msg = f"left and right images do not have same shape, got {stereo[0].shape}, {stereo[1].shape}"
assert stereo[0].shape == stereo[1].shape, msg

imleft = stereo[0]
imright = stereo[1]
imleft_color = stereo_color[0]
imright_color = stereo_color[1]

h, w = imright.shape #new height and with after undistortion

pts1, pts2 = findCorrespondence(imleft_color, imright_color, MATCHING, nMatches, DRAW_MATCHES)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

if FIND_FUNDAMENTAL:
######################################### FIND FUNDAMENTAL MATRIX  ########################################
    #at least 15 points are needed
    print("Finding fundamental matrix from matched points...\n")
    if MATCHING == MANUAL_MATCHING:
        F, mask = cv.findFundamentalMat(np.float32(pts1), np.float32(pts2), cv.FM_RANSAC)
    else:
        F, mask = cv.findFundamentalMat(np.float32(pts1), np.float32(pts2), cv.FM_LMEDS)

    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]

    print("Fundamental matrix: \n", F)

    #compute epilines
    l1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    l1 = l1.reshape(-1, 3)

    l2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    l2 = l2.reshape(-1, 3)

    if SHOW_EPILINES:
        epi_left, epi_leftc = drawlines(imleft, imright, l1, pts1, pts2)
        epi_right, epi_rightc = drawlines(imright, imleft, l2, pts2, pts1)

        fig, ax = plt.subplots(1, 2, figsize=(12,7))
        ax[0].imshow(epi_left)
        ax[1].imshow(epi_right)
        plt.show()

    #CHECK QUALITY USING EPIPOLAR CONSTRAINT
    checkEpipolarConst(pts1, pts2, l1, l2, F)

    # Get essential matrix from fundamental matrix and intrinsic parameters
    E = K[0].T @ F @ K[0]
    print(f"\nEssential matrix: \n{E}")

    E1 = K[1].T @ F @ K[1] # if K1 and K2 are the same E and E1 are the same
    print(f"\nEssential matrix (E1): \n{E1}")
else:
############################### FIND ESSENTIAL MATRIX DIRECTLY ####################################
    print("Finding essential matrix from matched points...\n")
    E, mask = cv.findEssentialMat(np.float32(pts1), np.float32(pts2), K[0], cv.RANSAC, 0.999, 1)
    print(f"Essential matrix (through findEssentialMatrix): \n{E}")

ret, R, t, _ = cv.recoverPose(E, pts1, pts2, K[0]) #recover relative camera rotation and translation

#performs chirality check to retrieve rotation such that 3d points lie in front of the camera
if USE_KITTI and FORCE_POSE:
    #FORCE pose from calibration file
    R_f = data['R2']
    t_f = data['T2']
    print(f"Estimated t:{t}")
    print("Forcing Rotation and translation from calibration file")
    print("Absolute difference recovered pose and \"true\" pose from calibration file:")
    print(f"\nRotation (error):\n{abs(R - R_f)}\nTranslation (error):\n{abs(t.ravel()-(t_f/np.linalg.norm(t_f)))}")
    R = R_f
    t = t_f

print(f'R:\n{R}')
print(f't:\n{t}')

euler = rotationMatrixToEulerAngles(R)
(R1rx, R1ry, R1rz) = np.rad2deg(euler)
print(f"\nEuler angles R:\nrx:{R1rx}, ry:{R1ry}, rz:{R1rz}\n")

###################################### STEREO RECTIFY ############################################
rectLeft, rectRight, projectionLeft, projectionRight, Q, roiL, roiR = cv.stereoRectify(K[0], dist[0], K[1], dist[1], 
                                                                                       (w_calib, h_calib), R, t, 
                                                                                       None, None, None, None, None, 
                                                                                       cv.CALIB_ZERO_DISPARITY, 0, (0, 0))

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

fname = "stereoData" + str(INDEX) + ".npz"
cv.imwrite('rectifiedLeft.jpg', rectifiedLeft)
cv.imwrite('rectifiedRight.jpg', rectifiedRight)
cv.imwrite('rectifiedLeft_c.jpg', rectifiedLeft_color)
cv.imwrite('rectifiedRight_c.jpg', rectifiedRight_color)

np.savez(fname, E=E, R=R, t=t, Q=Q)