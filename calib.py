import cv2
import numpy as np
import glob

# Checkerboard settings
checkerboard_size = (7, 10)
square_size = 3.3

# Arrays to store object points and image points
objpoints = []  # 3D points in real world space
imgpoints1 = []  # 2D points in image plane for camera 1
imgpoints2 = []  # 2D points in image plane for camera 2

# Prepare object points
objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) * square_size

# Load images
images1 = glob.glob('image/calib_*_1.png')
images2 = glob.glob('image/calib_*_2.png')

# Process each pair of images
for img1, img2 in zip(images1, images2):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret1, corners1 = cv2.findChessboardCorners(gray1, checkerboard_size, None)
    ret2, corners2 = cv2.findChessboardCorners(gray2, checkerboard_size, None)

    # If found, add object points and image points
    if ret1 and ret2:
        objpoints.append(objp)
        imgpoints1.append(corners1)
        imgpoints2.append(corners2)

# Load camera intrinsics
intrinsics1 = np.load('color_intrinsics_1.npy')
intrinsics2 = np.load('color_intrinsics_2.npy')

mtx1 = np.array([[intrinsics1[0], 0, intrinsics1[2]],
                 [0, intrinsics1[1], intrinsics1[3]],
                 [0, 0, 1]])

dist1 = intrinsics1[4:]

mtx2 = np.array([[intrinsics2[0], 0, intrinsics2[2]],
                 [0, intrinsics2[1], intrinsics2[3]],
                 [0, 0, 1]])

dist2 = intrinsics2[4:]

# Stereo calibration
ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints1, imgpoints2,
    mtx1, dist1, mtx2, dist2, gray1.shape[::-1],
    flags=cv2.CALIB_FIX_INTRINSIC
)

# Output results
if ret:
    print("Calibration successful")
    print("Rotation Matrix:\n", R)
    print("Translation Vector:\n", T)
    np.savez('calibration_data.npz', mtx1=mtx1, dist1=dist1, mtx2=mtx2, dist2=dist2, R=R, T=T)
else:
    print("Calibration failed")