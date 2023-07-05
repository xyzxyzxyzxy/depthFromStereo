# depthFromStereo
The objective of the project is to output the point clouds reconstructed from stereo images acquired with the same calibrated camera, simply by taking the first shot and moving the camera before taking a second shot of the same scene.

Currently the project is structured in 3 main python files that need to run in a specific order:

1. `intrinsicParameters.py sourceDirectory` to be called with the name of the directory containing the images you want to use to calibrate the camera. This outputs a `.npz` file containing the intrinsic parameters of the camera and distortion coefficients found by the camera calibration algorithm using the given images. 
(*Once the intrinsic parameters have been computed there is no need to run this again*)

2. `epipolarGeometry.py index` where _index_ will be the index of the stereo pair located by default in the directory `./stereoPairs/` (e.g use `epipolarGeometry.py 10` if you want to use stereo pair `./StereoPairs/L10.jpg ./StereoPairs/R10.jpg`). This writes the parameters that will be used to create the disparity to a file called `stereoDataX.npz` in the root directory where **X** is the index of the pair given as input.

    The file contains:

    1. The estimated fundamental matrix using matching pairs of keypoints found in the two images by SIFT.
    2. The essential matrix obtained from the fundamental matrix and the intrinsic paramters matrix (found by intrinsicParameters.py)
    3. Rotation matrix **R1** and translation vector **t** (obtained using SVD on the essential matrix)
    4. The (4x4) matrix **Q** used later for reprojection (see opencv documentation cv::reprojectTo3D and cv::stereoRectify)
    4. The dimensions of the Regions of Interest output to be applied to the rectified images
    
    \
    It also writes a rectified version of both shots in order to be able to perform stereo Block matching in a simpler way later. The four written files are two color rectified images of the two shots used, and their respective grayscale version. These are visible in the root directory as `rectified[Left | Right]_c.jpg` and `rectfified[Left | Right].jpg`.
    Sometimes rectification does not work with matrix **R1**, one of the two matrices returned by SVD, a solution can be to substitute **R1** with **R2**.

3. `depthMap.py index` where _index_ will be the index of the stereo pair used when calling `epipolarGeometry.py index`. This generates the disparity map with the aid of a window that allows to adjust the blockMatching algorithm parameters. When the parameters have been configured one can confirm by pressing *esc* on the keyboard. The disparity map produced is then used in conjunction with the matrix **Q** to produce a point cloud `stereoX.ply` where **X** is the index of the pair in exam.

    `depthMap.py path_to_left_img path_to_right_img` is an alternative way to run *depthMap.py* a disparity image is produced but not having matrix Q or any other parameter regarding the two cameras no reprojection is performed

If a new `.ply` file has to be generated for an existing `stereoDataX.npz` file It is necessary that the four images `rectified[Left | Right]_c.jpg` and `rectfified[Left | Right].jpg` in the root are the ones relative to the index of the pair the depth map has to be generated for. Otherwise `depthMap.py` will generate a depth map using images that may not correspond to the `stereoData.npz` file. In other wordrs always run `epipolarGeometry.py index` before `depthMap.py index` to avoid mismatches 