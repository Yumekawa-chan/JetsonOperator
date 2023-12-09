import cv2
import numpy as np
import glob

checkerboard_size = (7, 10) 
square_size = 1.0  

objpoints = [] 
imgpoints1 = [] 
imgpoints2 = []  

objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) * square_size

images1 = glob.glob('images/*_1.png')
images2 = glob.glob('images/*_2.png')

for img1, img2 in zip(images1, images2):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    ret1, corners1 = cv2.findChessboardCorners(gray1, checkerboard_size, None)
    ret2, corners2 = cv2.findChessboardCorners(gray2, checkerboard_size, None)

    if ret1 and ret2:
        objpoints.append(objp)
        imgpoints1.append(corners1)
        imgpoints2.append(corners2)

ret, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
    objpoints, imgpoints1, imgpoints2,
    None, None, None, None, gray1.shape[::-1],
    flags=cv2.CALIB_FIX_INTRINSIC
)

print("Rotation Matrix:")
print(R)
print("\nTranslation Vector:")
print(T)

calibration_data = {'R': R, 'T': T}

np.save('calibration_data.npy', calibration_data)