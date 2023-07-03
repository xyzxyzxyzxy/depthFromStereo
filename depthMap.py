import numpy as np
import cv2 as cv
import sys
from os.path import exists

REPROJECT_TO_3D = 1

if len(sys.argv) == 3:
    REPROJECT_TO_3D = 0
    fleft = sys.argv[1]
    fright = sys.argv[2]
    rectLeftRoi_color = cv.imread(fleft)
    rectRightRoi_color = cv.imread(fright)
    rectLeftRoi_color = cv.resize(rectLeftRoi_color,(0, 0),fx=0.15, fy=0.15, interpolation = cv.INTER_AREA)
    rectRightRoi_color = cv.resize(rectLeftRoi_color,(0, 0),fx=0.15, fy=0.15, interpolation = cv.INTER_AREA)
    rectLeftRoi = cv.cvtColor(rectLeftRoi_color, cv.COLOR_BGR2GRAY)
    rectRightRoi = cv.cvtColor(rectLeftRoi_color, cv.COLOR_BGR2GRAY)
elif len(sys.argv) > 1:
    INDEX = int(sys.argv[1])
    fname = "stereoData" + str(INDEX) + ".npz"

    if not exists(fname):
        print("epipolarGeometry.py needs to run first, ensure that the corresponding stereoData[idx].npz file has been generated.")
        sys.exit()

    data = np.load(fname)
    Q = data['Q']
    roiL = data['roiL']
    roiR = data['roiR']

    print(f"getting depth map from file: {fname}")

    print(f"Q:\n{Q}")
    print(f"roiL:\n{roiL}")
    print(f"roiR:\n{roiR}")

    xl, yl, wl, hl = roiL
    xr, yr, wr, hr = roiR

    rectifiedLeft = cv.imread('rectifiedLeft.jpg', cv.IMREAD_GRAYSCALE)
    rectifiedRight = cv.imread('rectifiedRight.jpg', cv.IMREAD_GRAYSCALE)
    rectifiedLeft_color = cv.imread('rectifiedLeft_c.jpg')
    rectifiedRight_color = cv.imread('rectifiedRight_c.jpg')
    # rectifiedLeft_color = cv.cvtColor(rectifiedLeft_color, cv.COLOR_BGR2RGB)
    # rectifiedRight_color = cv.cvtColor(rectifiedLeft_color, cv.COLOR_BGR2RGB)

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

    #make sure that the two rectified images have same size
    if rectLeftRoi.shape < rectRightRoi.shape:
        print("Left rectified smaller than right, resizing right")
        h = np.min([rectLeftRoi.shape[0], rectRightRoi.shape[0]])
        w = np.min([rectLeftRoi.shape[1], rectRightRoi.shape[1]])
        rectRightRoi = cv.resize(rectRightRoi, (w, h), cv.INTER_AREA)
        rectRightRoi_color = cv.resize(rectRightRoi_color, (w, h), cv.INTER_AREA)
    elif rectLeftRoi.shape > rectRightRoi.shape:
        print("Right rectified smaller than left, resizing left")
        h = np.min([rectLeftRoi.shape[0], rectRightRoi.shape[0]])
        w = np.min([rectLeftRoi.shape[1], rectRightRoi.shape[1]])
        rectLeftRoi = cv.resize(rectLeftRoi, (w, h), cv.INTER_AREA)
        rectLeftRoi_color = cv.resize(rectLeftRoi_color, (w, h), cv.INTER_AREA)
    else:
        pass
 
    print(f"size rectLeftRoi:\n{rectLeftRoi.shape}")
    print(f"size rectRightRoi:\n{rectRightRoi.shape}")
else:
    print("Either a stereo pair index or the path to two images has to be provided")
    sys.exit()

#STEREO BLOCK MATCHING USING STEREOBM
def nothing(x):
    pass
 
cv.namedWindow('disp',cv.WINDOW_NORMAL)
cv.resizeWindow('disp',600,80)
 
cv.createTrackbar('numDisparities','disp',1,15,nothing) #this is multiplied by 16
cv.createTrackbar('blockSize','disp',5,50,nothing)
cv.createTrackbar('preFilterType','disp',1,1,nothing)
cv.createTrackbar('preFilterSize','disp',2,25,nothing)
cv.createTrackbar('preFilterCap','disp',5,62,nothing)
cv.createTrackbar('textureThreshold','disp',10,100,nothing)
cv.createTrackbar('uniquenessRatio','disp',15,100,nothing)
cv.createTrackbar('speckleRange','disp',0,100,nothing)
cv.createTrackbar('speckleWindowSize','disp',3,25,nothing)
cv.createTrackbar('disp12MaxDiff','disp',5,25,nothing)
cv.createTrackbar('minDisparity','disp',5,25,nothing)

sbm = cv.StereoBM.create(numDisparities=64, blockSize=5)
disparity = 0

while True:
    disparity = sbm.compute(rectLeftRoi, rectRightRoi).astype(np.float32)/16.0

    numDisparities = cv.getTrackbarPos('numDisparities','disp')*16
    blockSize = cv.getTrackbarPos('blockSize','disp')*2 + 5
    preFilterType = cv.getTrackbarPos('preFilterType','disp')
    preFilterSize = cv.getTrackbarPos('preFilterSize','disp')*2 + 5
    preFilterCap = cv.getTrackbarPos('preFilterCap','disp')
    textureThreshold = cv.getTrackbarPos('textureThreshold','disp')
    uniquenessRatio = cv.getTrackbarPos('uniquenessRatio','disp')
    speckleRange = cv.getTrackbarPos('speckleRange','disp')
    speckleWindowSize = cv.getTrackbarPos('speckleWindowSize','disp')*2
    disp12MaxDiff = cv.getTrackbarPos('disp12MaxDiff','disp')
    minDisparity = cv.getTrackbarPos('minDisparity','disp')

    sbm.setNumDisparities(numDisparities)
    sbm.setBlockSize(blockSize)
    sbm.setPreFilterType(preFilterType)
    sbm.setPreFilterSize(preFilterSize)
    sbm.setPreFilterCap(preFilterCap)
    sbm.setTextureThreshold(textureThreshold)
    sbm.setUniquenessRatio(uniquenessRatio)
    sbm.setSpeckleRange(speckleRange)
    sbm.setSpeckleWindowSize(speckleWindowSize)
    sbm.setDisp12MaxDiff(disp12MaxDiff)
    sbm.setMinDisparity(minDisparity)

    cv.imshow('normalizedDisparity', (disparity-minDisparity)/numDisparities)
    if cv.waitKey(10) == 27:
        break

#project point cloud
accumulated_verts = None
def create_pointcloud(disparity, Q, color):
    im3d = cv.reprojectImageTo3D(disparity ,Q, handleMissingValues=True)

    PLY_HEADER = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

    FILENAME = "cloud" +str(INDEX) + ".ply"

    def write_ply():
        with open(FILENAME, 'w') as f:
            f.write(PLY_HEADER % dict(vert_num=len(accumulated_verts)))
            np.savetxt(f, accumulated_verts, '%f %f %f %d %d %d')

    def append_ply_array(verts, colors):
        global accumulated_verts
        verts = verts.reshape(-1, 3)
        colors = colors.reshape(-1, 3)
        verts_new = np.hstack([verts, colors])
        accumulated_verts = verts_new

    mask = disparity > disparity.min()
    out_points = im3d[mask]
    out_colors = color[mask]
    append_ply_array(out_points, out_colors)
    write_ply()

if REPROJECT_TO_3D:
    color = rectLeftRoi_color #whatever left or rigtht is good ROI
    create_pointcloud(disparity, Q, color)