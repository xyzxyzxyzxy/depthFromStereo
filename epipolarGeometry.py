import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math

def findCorrespondence(imleft, imright, matching=False, nMatches=100, draw=True):

    imleft_color = imleft
    imright_color = imright

    imleft = cv.cvtColor(imleft_color, cv.COLOR_RGB2GRAY)
    imright = cv.cvtColor(imright_color, cv.COLOR_RGB2GRAY)

    #use sift to get keypoints and descriptors
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(imleft, None)
    kp2, des2 = sift.detectAndCompute(imright, None)

    pts1 = []
    pts2 = []

    if matching == 1 and nMatches > 0:
        #USE BRUTEFORCE MATCHING
        print("\nUSING BRUTEFORCE MATCHING\n")
        bf = cv.BFMatcher()
        matches = bf.match(des1,des2)
        matches = sorted(matches, key = lambda x :x.distance)
        chosen_matches = matches[0:nMatches]
        for i, m in enumerate(chosen_matches):
            #print(f"i: {i}, n: {n}, m: {m}")
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
        if draw:
            matched_image = cv.drawMatches(imleft,kp1,imright,kp2,matches[:nMatches],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            plt.figure(figsize=(12,7))
            plt.imshow(matched_image)
            plt.show()
    elif matching == 0:
        #USE FLANN MATCHER
        print("\nUSING FLANN MATCHING\n")
        FLANN_INDEX_KDTREE = 1
        index_par = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_par = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_par, search_par)
        matches = flann.knnMatch(des1, des2, k=2)

        ratio_thresh = 0.8 #Lowe's ratio thresh, matching is very sensitive to this value

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
        if draw:
            plt.figure(figsize=(12,7))
            im_matches = cv.drawMatchesKnn(imleft,kp1,imright,kp2,matches,None,**draw_params)
            plt.imshow(im_matches,)
            plt.show()
    else:
        print("\nUSING MANUAL MATCHING\n")
        imleft_copy = np.copy(imleft_color)
        imright_copy = np.copy(imright_color)
        for i in range(15): #find 15
            pts1 = selectKeypoint(imleft_copy, "left", pts1, i)
            pts2 = selectKeypoint(imright_copy, "right", pts2, i)
            cv.destroyAllWindows()
        if draw:
            plt.figure(figsize=(12,7))
            im_matches = drawMatches(imleft_color, pts1, imright_color, pts2)
            plt.imshow(im_matches,)
            plt.show()

    return pts1, pts2

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

def checkEpipolarConst(pts1, pts2, l1, l2, F):
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

def selectKeypoint(img, name, pts, index):
    # callback function
    index = index + 1
    name = name + str(index)
    c = 0
    def click_event(event, x, y, flags, params):
        # checking for left mouse clicks
        nonlocal c
        if event == cv.EVENT_LBUTTONDOWN and c == 0:
            c = c + 1
            # displaying the coordinates
            # on the Shell

            pt = (x, y)
            print(pt)
            pts.append(pt)

            # displaying the coordinates
            # on the image window

            color = ((0 + 70*index) % 255 , (200 + 50*index) % 255, (0 + 20*index) % 255)
            
            font = 1
            cv.putText(img, str(x) + ',' +
                        str(y), (x,y), font,
                        1, color, 1)
            
            cv.circle(img, (x, y), radius=2, color=color, thickness=3)
            cv.imshow(name, img)
    
        # checking for right mouse clicks     
        # if event==cv.EVENT_RBUTTONDOWN:
    
        #     # displaying the coordinates
        #     # on the Shell
        #     print(x, ' ', y)
    
        #     # displaying the coordinates
        #     # on the image window
        #     font = 1
        #     b = img[y, x, 0]
        #     g = img[y, x, 1]
        #     r = img[y, x, 2]
        #     cv.putText(img, str(b) + ',' +
        #                 str(g) + ',' + str(r),
        #                 (x,y), font, 1,
        #                 (255, 255, 255), 1)
        #     cv.circle(img, (x, y), radius=2, color=(255, 255, 255), thickness=-1)
        #     cv.imshow(name, img)

    cv.imshow(name, img)
    cv.setMouseCallback(name, click_event)
    cv.waitKey(0)
    return pts

def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
 
    assert(isRotationMatrix(R))
 
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
 
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])

def drawMatches(imleft, pts1, imright, pts2):
    im_matches = np.concatenate((imleft, imright), axis=1)
    h, w = imleft.shape[:2]
    for i in range(len(pts1)):
        color = tuple(np.random.randint(0,255,3).tolist())
        cv.circle(im_matches, (pts1[i][0], pts1[i][1]), 2, color, 5)
        cv.circle(im_matches, (pts2[i][0] + w, pts2[i][1]), 2, color, 5)
        cv.line(im_matches, (pts1[i][0], pts1[i][1]), (pts2[i][0] + w, pts2[i][1]) , color, 3)

    return cv.cvtColor(im_matches, cv.COLOR_BGR2RGB)
