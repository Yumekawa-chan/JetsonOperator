import cv2
import cv2.aruco as aruco
import numpy as np
from glob import glob

id_1 = input("行列を適用するのは？: ")
id_2 = input("動かさないほうは？: ")

def find_aruco_markers(image, marker_size=6, total_markers=250, draw=True):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{marker_size}X{marker_size}_{total_markers}')
    aruco_dict = aruco.getPredefinedDictionary(key)
    aruco_params = aruco.DetectorParameters()
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    
    if draw:
        aruco.drawDetectedMarkers(image, corners, ids)

    return corners, ids

def estimate_pose(image, corners, ids, camera_matrix, dist_coeff):
    if corners and len(corners) > 0 and ids is not None and len(ids) > 0:
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.39, camera_matrix, dist_coeff)
        for rvec, tvec in zip(rvecs, tvecs):
            cv2.drawFrameAxes(image, camera_matrix, dist_coeff, rvec, tvec, 0.04)
        return rvecs, tvecs
    return None, None

def relative_camera_pose(rvec1, tvec1, rvec2, tvec2):
    R1, _ = cv2.Rodrigues(rvec1)
    R2, _ = cv2.Rodrigues(rvec2)
    R_rel = np.dot(R2, R1.T)
    t_rel = tvec2 - np.dot(R_rel, tvec1)
    return R_rel, t_rel

def save_rt_matrices(R, T):
    np.save(f'matrix/R_{id_1}_{id_2}.npy', R)
    np.save(f'matrix/T_{id_1}_{id_2}.npy', T)

camera_matrix_1 = np.load(f"matrix/camera_matrix_{id_1}.npy")
dist_coeff_1 = np.load(f"matrix/dist_coeff_{id_1}.npy")
camera_matrix_2 = np.load(f"matrix/camera_matrix_{id_2}.npy")
dist_coeff_2 = np.load(f"matrix/dist_coeff_{id_2}.npy")

print("camera_matrix_1:\n",camera_matrix_1)
print("dist_coeff_1:\n",dist_coeff_1)
print("camera_matrix_2:\n",camera_matrix_2)
print("dist_coeff_2:\n",dist_coeff_2)

image1_paths = glob(f'image/color_*_{id_1}.png')
image2_paths = glob(f'image/color_*_{id_2}.png')

for image1_path, image2_path in zip(image1_paths, image2_paths):
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    corners1, ids1 = find_aruco_markers(image1)
    corners2, ids2 = find_aruco_markers(image2)

    if ids1 is not None and ids2 is not None:
        rvecs1, tvecs1 = estimate_pose(image1, corners1, ids1, camera_matrix_1, dist_coeff_1)
        rvecs2, tvecs2 = estimate_pose(image2, corners2, ids2, camera_matrix_2, dist_coeff_2)

        common_ids = np.intersect1d(ids1.flatten(), ids2.flatten())

        for id in common_ids:
            idx1 = np.where(ids1 == id)[0][0]
            idx2 = np.where(ids2 == id)[0][0]
            R, T = relative_camera_pose(rvecs1[idx1], tvecs1[idx1][0], rvecs2[idx2], tvecs2[idx2][0])
            print(f"マーカー {id} のカメラ1からカメラ2への相対的なR: \n{R}")
            print(f"マーカー {id} のカメラ1からカメラ2への相対的なT: \n{T}")

            save_rt_matrices(R, T)

        cv2.imshow("Image 1", image1)
        cv2.imshow("Image 2", image2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()