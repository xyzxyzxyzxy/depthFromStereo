import numpy as np
import cv2 as cv
import sys
from os.path import exists

REPROJECT_TO_3D = 1

if len(sys.argv) == 3:
    REPROJECT_TO_3D = 0
    fleft = sys.argv[1]
    fright = sys.argv[2]
    rectifiedLeft_color = cv.imread(fleft)
    rectifiedRight_color = cv.imread(fright)
    #rectifiedLeft_color = cv.resize(rectifiedLeft_color,(0, 0),fx=0.15, fy=0.15, interpolation = cv.INTER_AREA)
    #rectifiedRight_color = cv.resize(rectifiedRight_color,(0, 0),fx=0.15, fy=0.15, interpolation = cv.INTER_AREA)
    rectifiedLeft = cv.cvtColor(rectifiedLeft_color, cv.COLOR_BGR2GRAY)
    rectifiedRight = cv.cvtColor(rectifiedRight_color, cv.COLOR_BGR2GRAY)
elif len(sys.argv) > 1:
    INDEX = int(sys.argv[1])
    fname = "stereoData" + str(INDEX) + ".npz"

    if not exists(fname):
        print("rectifyStereoPair.py needs to run first, ensure that the corresponding stereoData[idx].npz file has been generated.")
        sys.exit()

    data = np.load(fname)
    Q = data['Q']
    
    print(f"getting depth map from file: {fname}")

    print(f"Q:\n{Q}")

    rectifiedLeft = cv.imread('rectifiedLeft.jpg', cv.IMREAD_GRAYSCALE)
    rectifiedRight = cv.imread('rectifiedRight.jpg', cv.IMREAD_GRAYSCALE)
    rectifiedLeft_color = cv.imread('rectifiedLeft_c.jpg')
    rectifiedRight_color = cv.imread('rectifiedRight_c.jpg')
    rectifiedLeft_color = cv.cvtColor(rectifiedLeft_color, cv.COLOR_BGR2RGB)
    rectifiedRight_color = cv.cvtColor(rectifiedLeft_color, cv.COLOR_BGR2RGB)
else:
    print("Either a stereo pair index or the path to two images has to be provided")
    sys.exit()

#STEREO BLOCK MATCHING USING STEREOBM
def nothing(x):
    pass
 
cv.namedWindow('disp',cv.WINDOW_NORMAL)
cv.resizeWindow('disp',600,150)
 
cv.createTrackbar('numDisparities','disp',1,25,nothing) #this is multiplied by 16
cv.createTrackbar('blockSize','disp',5,50,nothing)
cv.createTrackbar('preFilterType','disp',1,1,nothing)
cv.createTrackbar('preFilterSize','disp',2,25,nothing)
cv.createTrackbar('preFilterCap','disp',5,62,nothing)
cv.createTrackbar('textureThreshold','disp',10,100,nothing)
cv.createTrackbar('uniquenessRatio','disp',15,100,nothing)
cv.createTrackbar('speckleRange','disp',0,100,nothing)
cv.createTrackbar('speckleWindowSize','disp',3,25,nothing)
cv.createTrackbar('disp12MaxDiff','disp',5,25,nothing)
cv.createTrackbar('minDisparity','disp',0,20,nothing)

sbm = cv.StereoBM.create(numDisparities=64, blockSize=5)
disparity = 0

while True:
    disparity = sbm.compute(rectifiedLeft, rectifiedRight).astype(np.float32)/16.0

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


    #disp = ((disparity-minDisparity)/numDisparities)
    disp = cv.normalize(disparity, None, 255, 0, cv.NORM_MINMAX, cv.CV_8U)
    disp = cv.applyColorMap(disp, cv.COLORMAP_INFERNO)
    cv.imshow('normalizedDisparity', disp)
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
    color = rectifiedLeft_color #whatever left or rigtht is good
    create_pointcloud(disparity, Q, color)