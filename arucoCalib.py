import cv2
import numpy as np
from glob import glob

def detect_aruco_and_calibrate(image_file1, image_file2, intrinsics_file1, intrinsics_file2):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters()

    img1 = cv2.imread(image_file1)
    img2 = cv2.imread(image_file2)

    camera_matrix_1, dist_coeffs_1 = load_camera_parameters(intrinsics_file1)
    camera_matrix_2, dist_coeffs_2 = load_camera_parameters(intrinsics_file2)

    corners1, ids1, _ = cv2.aruco.detectMarkers(img1, aruco_dict, parameters=aruco_params)
    corners2, ids2, _ = cv2.aruco.detectMarkers(img2, aruco_dict, parameters=aruco_params)

    img1_markers = cv2.aruco.drawDetectedMarkers(img1.copy(), corners1, ids1)
    img2_markers = cv2.aruco.drawDetectedMarkers(img2.copy(), corners2, ids2)

    # 画像を表示
    cv2.imshow('Image 1 with Markers', img1_markers)
    cv2.imshow('Image 2 with Markers', img2_markers)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if ids1 is None or ids2 is None:
        return None, None 
    common_ids = np.intersect1d(ids1, ids2)

    marker_size = 0.84   # マーカーのサイズ[m]
    marker_corners_3d = np.array([
        [-marker_size / 2, marker_size / 2, 0],
        [marker_size / 2, marker_size / 2, 0],
        [marker_size / 2, -marker_size / 2, 0],
        [-marker_size / 2, -marker_size / 2, 0]
    ], dtype=np.float32) 

    for id in common_ids:
        idx1 = np.where(ids1 == id)[0]
        idx2 = np.where(ids2 == id)[0]

        if idx1.size > 0 and idx2.size > 0:
            idx1 = idx1[0]
            idx2 = idx2[0]

            corners1_2d = np.array(corners1[idx1][0], dtype=np.float32).reshape(-1, 1, 2)
            corners2_2d = np.array(corners2[idx2][0], dtype=np.float32).reshape(-1, 1, 2)

            ret1, rvec1, tvec1 = cv2.solvePnP(marker_corners_3d, corners1_2d, camera_matrix_1, dist_coeffs_1)
            ret2, rvec2, tvec2 = cv2.solvePnP(marker_corners_3d, corners2_2d, camera_matrix_2, dist_coeffs_2)
        

            if ret1 and ret2:
                R1, _ = cv2.Rodrigues(rvec1)
                R2, _ = cv2.Rodrigues(rvec2)

                R_relative = np.dot(R2.T, R1) 
                t_relative = np.dot(R2.T, tvec1 - tvec2) 
                return R_relative, t_relative

    return None, None 

def load_camera_parameters(intrinsics_file):
    params = np.load(intrinsics_file)

    camera_matrix = np.array([[params[0], 0, params[2]],
                              [0, params[1], params[3]],
                              [0, 0, 1]])

    dist_coeffs = params[4:9].reshape(-1, 1)

    return camera_matrix, dist_coeffs

image_file1 = glob('image\color_*1.png')[0]
image_file2 = glob('image\color_*_2.png')[0]
intrinsics_file1 = 'matrix\color_intrinsics_1.npy'
intrinsics_file2 = 'matrix\color_intrinsics_2.npy'

R_relative, t_relative = detect_aruco_and_calibrate(image_file1, image_file2, intrinsics_file1, intrinsics_file2)

if R_relative is not None and t_relative is not None:
    print("相対回転行列\n", R_relative)
    print("相対並進ベクトル:\n", t_relative)
    np.save("matrix/R.npy", R_relative)
    np.save("matrix/T.npy", t_relative)
else:
    print("共通のArucoマーカーが検出されませんでした。")
