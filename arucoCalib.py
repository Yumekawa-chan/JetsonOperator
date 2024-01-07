import cv2
import cv2.aruco as aruco
import numpy as np

def find_aruco_markers(image, marker_size=6, total_markers=250, draw=True):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{marker_size}X{marker_size}_{total_markers}')
    aruco_dict = aruco.Dictionary_get(key)
    aruco_params = aruco.DetectorParameters_create()
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    
    if draw:
        aruco.drawDetectedMarkers(image, corners, ids)

    return corners, ids

def estimate_pose(image, corners, ids, camera_matrix, dist_coeff):
    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeff)
    for rvec, tvec in zip(rvecs, tvecs):
        aruco.drawAxis(image, camera_matrix, dist_coeff, rvec, tvec, 0.03)
    return rvecs, tvecs

def relative_camera_pose(rvec1, tvec1, rvec2, tvec2):
    R1, _ = cv2.Rodrigues(rvec1)
    R2, _ = cv2.Rodrigues(rvec2)
    R_rel = np.dot(R2, R1.T)
    t_rel = tvec2 - np.dot(R_rel, tvec1)
    return R_rel, t_rel

# カメラ内部パラメータ（これらはあなたのカメラに合わせて設定する必要があります）
camera_matrix = np.array([[800, 0, 320],
                          [0, 800, 240],
                          [0, 0, 1]])
dist_coeff = np.zeros(4)

# 画像を読み込む
image1 = cv2.imread('path_to_image1.jpg')
image2 = cv2.imread('path_to_image2.jpg')

# マーカーを検出し、ポーズを推定する
corners1, ids1 = find_aruco_markers(image1)
corners2, ids2 = find_aruco_markers(image2)
rvecs1, tvecs1 = estimate_pose(image1, corners1, ids1, camera_matrix, dist_coeff)
rvecs2, tvecs2 = estimate_pose(image2, corners2, ids2, camera_matrix, dist_coeff)

# 共通マーカーのIDを見つける
common_ids = np.intersect1d(ids1.flatten(), ids2.flatten())

# RとTの行列を計算する
for id in common_ids:
    idx1 = np.where(ids1 == id)[0][0]
    idx2 = np.where(ids2 == id)[0][0]
    R, T = relative_camera_pose(rvecs1[idx1], tvecs1[idx1][0], rvecs2[idx2], tvecs2[idx2][0])
    print(f"マーカー {id} のカメラ1からカメラ2への相対的なR: \n{R}")
    print(f"マーカー {id} のカメラ1からカメラ2への相対的なT: \n{T}")
