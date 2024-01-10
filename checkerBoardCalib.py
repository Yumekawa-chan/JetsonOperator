import numpy as np
import cv2
from glob import glob

# Checkerboard configuration
CHECKERBOARD = (4, 7)  # チェッカーボードの内側の角の数
square_size = 0.1     # チェッカーボードの各正方形のサイズ（単位：センチメートル）

# Calibration criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 3D object points preparation
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store object points and image points
objpoints = []  # 3D points in real world space
imgpoints1 = [] # 2D points in image plane for camera 1
imgpoints2 = [] # 2D points in image plane for camera 2

# Load camera parameters
def load_camera_parameters(filepath):
    data = np.load(filepath)
    mtx = np.array([[data[0], 0, data[2]],
                    [0, data[1], data[3]],
                    [0, 0, 1]])
    dist = data[4:]
    return mtx, dist

def save_rt_matrices(R, T):
    np.save(f'matrix/R.npy', R)
    np.save(f'matrix/T.npy', T)

mtx1, dist1 = load_camera_parameters('matrix/color_intrinsics_1.npy')
mtx2, dist2 = load_camera_parameters('matrix/color_intrinsics_2.npy')

# Image file paths
image_folder = 'image/'
image_files1 = sorted(glob(image_folder + 'color_*_1.png'))
image_files2 = sorted(glob(image_folder + 'color_*_2.png'))

# Iterate over pairs of images
for img_file1, img_file2 in zip(image_files1, image_files2):
    img1 = cv2.imread(img_file1)
    img2 = cv2.imread(img_file2)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret1, corners1 = cv2.findChessboardCorners(gray1, CHECKERBOARD, None)
    ret2, corners2 = cv2.findChessboardCorners(gray2, CHECKERBOARD, None)

    if ret1 and ret2:
        objpoints.append(objp)
        corners1 = cv2.cornerSubPix(gray1, corners1, (11,11), (-1,-1), criteria)
        corners2 = cv2.cornerSubPix(gray2, corners2, (11,11), (-1,-1), criteria)
        imgpoints1.append(corners1)
        imgpoints2.append(corners2)

# Stereo calibration
ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints1, imgpoints2, mtx1, dist1, mtx2, dist2, gray1.shape[::-1], 
    criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC
)

print("Stereo Calibration successful:", ret)
print("Rotation Matrix:\n", R)
print("Translation Vector:\n", T)

save_rt_matrices(R, T)
