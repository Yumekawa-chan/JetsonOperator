import cv2
import cv2.aruco as aruco
import numpy as np
from glob import glob


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
        
    return rvecs, tvecs

def relative_camera_pose(rvec1, tvec1, rvec2, tvec2):
    R1, _ = cv2.Rodrigues(rvec1)
    R2, _ = cv2.Rodrigues(rvec2)
    R_rel = np.dot(R2, R1.T)

    tvec1 = tvec1.reshape((3, 1))
    tvec2 = tvec2.reshape((3, 1))

    t_rel = tvec2 - np.dot(R_rel, tvec1)
    return R_rel, t_rel.reshape((3, 1))



def load_camera_parameters(filepath):
    intrinsics = np.load(filepath)
    camera_matrix = np.array([[intrinsics[0], 0, intrinsics[2]],
                              [0, intrinsics[1], intrinsics[3]],
                              [0, 0, 1]])
    if len(intrinsics) == 9:
        dist_coeff = intrinsics[4:9]
    else:
        dist_coeff = np.append(intrinsics[4:], [0])
    return camera_matrix, dist_coeff

def save_rt_matrices(R, T):
    np.save(f'matrix/R.npy', R)
    np.save(f'matrix/T.npy', T)

def calculate_relative_camera_pose(image1, image2, camera_matrix_1, dist_coeff_1, camera_matrix_2, dist_coeff_2):
    corners1, ids1 = find_aruco_markers(image1)
    corners2, ids2 = find_aruco_markers(image2)

    R_list = []
    T_list = []

    if ids1 is not None and ids2 is not None:
        rvecs1, tvecs1 = estimate_pose(image1, corners1, ids1, camera_matrix_1, dist_coeff_1)
        rvecs2, tvecs2 = estimate_pose(image2, corners2, ids2, camera_matrix_2, dist_coeff_2)

        common_ids = np.intersect1d(ids1.flatten(), ids2.flatten())

        for id in common_ids:
            idx1 = np.where(ids1 == id)[0][0]
            idx2 = np.where(ids2 == id)[0][0]
            R_rel, t_rel = relative_camera_pose(rvecs1[idx1], tvecs1[idx1], rvecs2[idx2], tvecs2[idx2])
            R_list.append(R_rel)
            T_list.append(t_rel)

    if R_list and T_list:
        R_avg = np.mean(R_list, axis=0)
        T_avg = np.mean(T_list, axis=0)
        return R_avg, T_avg
    else:
        return None, None

# メイン処理
camera_matrix_1, dist_coeff_1 = load_camera_parameters('matrix/color_intrinsics_1.npy')
camera_matrix_2, dist_coeff_2 = load_camera_parameters('matrix/color_intrinsics_2.npy')

image1_paths = glob('image/color_*_1.png')
image2_paths = glob('image/color_*_2.png')

for image1_path, image2_path in zip(image1_paths, image2_paths):
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    R, T = calculate_relative_camera_pose(image1, image2, camera_matrix_1, dist_coeff_1, camera_matrix_2, dist_coeff_2)

    if R is not None and T is not None:
        print(f"カメラ間の平均相対的なR: \n{R}")
        print(f"カメラ間の平均相対的なT: \n{T}")
        save_rt_matrices(R, T)

        cv2.imshow("Image 1", image1)
        cv2.imshow("Image 2", image2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()