import numpy as np
import glob
import cv2 as cv
import undistort
import matplotlib.pyplot as plt
import sys

SHOW_EPILINES = 1
CORRECT_LENS_DISTORTION = 1
CHECK_EPIPOLAR_CONSTRAINT = 1
DRAW_MATCHES = 1
SHOW_ROIS = 0
BRUTE_FORCE_MATCHING = 0 # IF SET TO 0 FLANN IS USED

if len(sys.argv) < 2:
    print("Stereo pair index required")
    sys.exit()

INDEX = int(sys.argv[1])

#load stereo pair with index INDEX
filename = "./StereoPairs/*{:02d}.JPG".format(INDEX)
print(filename)
stereo_pair = glob.glob(filename)

print(f"Loading stereo pair: {stereo_pair}")
print(f"Loading intrinsic parameters")

data = np.load("intrinsicParameters.npz")
K = data['K']
dist = data['dist']
w = int(data['w'])
h = int(data['h'])

print("\nIntrinsic parameter matrix: \n", K)
print("\nDistortion coefficients: \n", dist)
print("Image_size: \n", w, h)

stereo = []
stereo_color = []

for fname in stereo_pair:
    img_color = cv.imread(fname)
    img_color = cv.cvtColor(img_color, cv.COLOR_BGR2RGB)
    img_color = cv.resize(img_color, (w, h), interpolation = cv.INTER_AREA)
    img_gray = cv.cvtColor(img_color, cv.COLOR_RGB2GRAY)
    if CORRECT_LENS_DISTORTION:
        undist, newmat = undistort.undistort(img_gray, K, dist)
        print(f"Shape after undistortion using intrinsic parameters: {undist.shape}")
    stereo.append(undist)
    stereo_color.append(img_color) #Not undistorted
    cv.imshow(fname, undist)
    cv.waitKey(0)


K = newmat
print("\nNew Intrinsic parameter matrix: \n", K)

imright = stereo[0]
imleft = stereo[1]
imright_color = stereo_color[0]
imleft_color = stereo_color[1]

h, w = imright.shape #new height and with of both images

print(f"imright type: {imright.dtype}")

#use sift to get keypoints and descriptors
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(imleft, None)
kp2, des2 = sift.detectAndCompute(imright, None)

pts1 = []
pts2 = []

if BRUTE_FORCE_MATCHING:
    bf = cv.BFMatcher()
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x :x.distance)
    chosen_matches = matches[0:100]
    
    for i, m in enumerate(chosen_matches):
        #print(f"i: {i}, n: {n}, m: {m}")
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
    
    if DRAW_MATCHES:
        matched_image = cv.drawMatches(imleft,kp1,imright,kp2,matches[:100],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.figure(figsize=(12,7))
        plt.imshow(matched_image)
        plt.show()
else:
    #find keypoints matches using FLANN MATCHER
    FLANN_INDEX_KDTREE = 1
    index_par = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_par = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_par, search_par)
    matches = flann.knnMatch(des1, des2, k=2)

    ratio_thresh = 0.8

    matchesMask = [[0,0] for i in range(len(matches))]

    draw_params = dict(
        #matchColor = (0,255,80),
        #singlePointColor = (120,0,255),
        matchesMask = matchesMask,
        flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    for i, (m, n) in enumerate(matches):
        #print(f"i: {i}, n: {n}, m: {m}")
        if m.distance < ratio_thresh*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            matchesMask[i] = [1, 0]

    if DRAW_MATCHES:
        plt.figure(figsize=(12,7))
        im_matches = cv.drawMatchesKnn(imleft,kp1,imright,kp2,matches,None,**draw_params)
        plt.imshow(im_matches,)
        plt.show()

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

#F, mask = cv.findFundamentalMat(pts1 ,pts2, cv.FM_RANSAC, 3, 0.99) 
F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)

pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

print("\nFundamental matrix: \n", F)
#print("points1: \n", pts1.reshape(-1, 1, 2), "\npoints2: \n", pts2.reshape(-1, 1, 2))
#print(len(pts1), len(pts2))

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
    lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

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
    ptl = np.copy(pts1)
    ptr = np.copy(pts2)
    ptl = np.concatenate((ptl, np.ones((ptl.shape[0], 1))), 1)
    ptr = np.concatenate((ptr, np.ones((ptr.shape[0], 1))), 1)

    error = 0
    for i in range(ptl.shape[0]):
        error = error + np.abs(ptl[i, :] @ F @ ptr[i, :].T) + np.abs(ptr[i, :] @ F @ ptl[i, :].T)
    print(f"\nTotal error: {error/((ptl.shape[0]))}")

    error_epi = 0
    for i in range(ptl.shape[0]):
        error_epi = error_epi + np.abs(ptl[i, 0]* l2[i, 0] + ptl[i, 1]* l2[i, 1] + ptl[i, 2] * l2[i, 2]) +\
            np.abs(ptr[i, 0]* l1[i, 0] + ptr[i, 1]* l1[i, 1] + ptr[i, 2] * l1[i, 2])
    print(f"\nTotal error epilines: {error_epi/((ptl.shape[0]))}")

#get essential matrix from fundamental matrix
E = K.T @ F @ K
print(f"\nEssential matrix: \n{E}")

R1, R2, t, = cv.decomposeEssentialMat(E)
print(f"\nR1: {R1}, \nR2: {R2}, \nt: {t}")

#DEBUG
# print(t.shape)
# print(R1.shape)
# t[1:3] = 0
# t[0] = 1
# R1 = np.eye(3)
# print("DEBUG T: ", t)
# print("DEBUG R1: ", R1)

#STEREORECTIFY
rectLeft, rectRight, projectionLeft, projectionRight, Q, roiL, roiR = cv.stereoRectify(K, dist, K, dist, 
                                                                                       (w, h), R2, t, 
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

mapxL, mapyL = cv.initUndistortRectifyMap(K, dist, rectLeft, projectionLeft, (w, h), cv.CV_32FC1)
mapxR, mapyR = cv.initUndistortRectifyMap(K, dist, rectRight, projectionRight, (w, h), cv.CV_32FC1)

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