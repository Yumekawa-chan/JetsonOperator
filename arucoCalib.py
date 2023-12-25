import cv2
import cv2.aruco as aruco
import numpy as np
import glob

def detect_aruco_markers(image, aruco_dict_type=aruco.DICT_6X6_250, marker_size=0.05, camera_matrix=None, dist_coeffs=None):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco_dict_type)
    aruco_params = aruco.DetectorParameters_create()
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params, cameraMatrix=camera_matrix, distCoeff=dist_coeffs)
    
    if len(corners) > 0:
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)
        for i in range(len(ids)):
            image = aruco.drawAxis(image, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.1)
        aruco.drawDetectedMarkers(image, corners, ids)
    return image, corners, ids, rvecs, tvecs

def get_transformation_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.array(tvec).reshape(3, 1)
    transformation_matrix = np.concatenate((R, T), axis=1)
    transformation_matrix = np.vstack((transformation_matrix, [0, 0, 0, 1]))
    return transformation_matrix

# カメラの内部パラメータ（キャリブレーションから取得）
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])  # ここに実際の値を設定
dist_coeffs = np.array([k1, k2, p1, p2, k3])  # ここに実際の値を設定

# 画像の読み込みとArUcoマーカーの検出
filepaths1 = glob.glob('path/to/timestamp_1_*.png')  # 画像のパスを正しく設定
filepaths2 = glob.glob('path/to/timestamp_2_*.png')  # 画像のパスを正しく設定

for filepath1, filepath2 in zip(filepaths1, filepaths2):
    image1 = cv2.imread(filepath1)
    image2 = cv2.imread(filepath2)

    # それぞれのカメラでArUcoマーカーを検出
    image_with_markers1, corners1, ids1, rvecs1, tvecs1 = detect_aruco_markers(image1, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    image_with_markers2, corners2, ids2, rvecs2, tvecs2 = detect_aruco_markers(image2, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

    if len(rvecs1) > 0 and len(rvecs2) > 0:
        # 最初のマーカーの姿勢を変換行列に変換
        transformation_matrix1 = get_transformation_matrix(rvecs1[0], tvecs1[0])
        transformation_matrix2 = get_transformation_matrix(rvecs2[0], tvecs2[0])

        print("Transformation Matrix for Camera 1:")
        print(transformation_matrix1)
        print("\nTransformation Matrix for Camera 2:")
        print(transformation_matrix2)

    # 結果の表示（オプション）
    cv2.imshow('Camera 1 ArUco Markers', image_with_markers1)
    cv2.imshow('Camera 2 ArUco Markers', image_with_markers2)
    cv2.waitKey(0)

cv2.destroyAllWindows()
