import cv2
import cv2.aruco as aruco
import numpy as np
from glob import glob


def find_best_pose(poses, corner_points):
    best_score = np.inf
    best_index = -1
    
    for i, pose in enumerate(poses):
        distance_squared = 0
        for corner_point in corner_points:
            transformed_point = pose @ np.append(corner_point, 1)
            distance_squared += np.sum((transformed_point[:-1] - corner_point) ** 2)
        
        if distance_squared < best_score:
            best_score = distance_squared
            best_index = i
            
    return best_index

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
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.3, camera_matrix, dist_coeff)
        for rvec, tvec in zip(rvecs, tvecs):
            cv2.drawFrameAxes(image, camera_matrix, dist_coeff, rvec, tvec, 0.3)
        return rvecs, tvecs
    return None, None

def relative_camera_pose(rvec1, tvec1, rvec2, tvec2):
    R1, _ = cv2.Rodrigues(rvec1)
    R2, _ = cv2.Rodrigues(rvec2)
    R_rel = np.dot(R2, R1.T)
    t_rel = tvec2 - np.dot(R_rel, tvec1)
    return R_rel, t_rel

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

camera_matrix_1, dist_coeff_1 = load_camera_parameters('matrix/color_intrinsics_1.npy')
camera_matrix_2, dist_coeff_2 = load_camera_parameters('matrix/color_intrinsics_2.npy')

image1_paths = glob('image/color_*_1.png')
image2_paths = glob('image/color_*_2.png')

all_poses1, all_poses2 = [], []
all_corners1, all_corners2 = [], []
all_ids1, all_ids2 = [], []

for image1_path, image2_path in zip(image1_paths, image2_paths):
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    corners1, ids1 = find_aruco_markers(image1)
    corners2, ids2 = find_aruco_markers(image2)

    rvecs1, tvecs1 = estimate_pose(image1, corners1, ids1, camera_matrix_1, dist_coeff_1)
    rvecs2, tvecs2 = estimate_pose(image2, corners2, ids2, camera_matrix_2, dist_coeff_2)

    all_poses1.append((rvecs1, tvecs1))
    all_poses2.append((rvecs2, tvecs2))
    all_corners1.append(corners1)
    all_corners2.append(corners2)
    all_ids1.append(ids1)
    all_ids2.append(ids2)

common_ids = np.array([id for ids in all_ids1 for id in ids.flatten()])
common_ids = np.intersect1d(common_ids, np.array([id for ids in all_ids2 for id in ids.flatten()]))

for common_id in common_ids:
    poses1 = []
    poses2 = []
    corner_points = []

    for i, ids in enumerate(all_ids1):
        if common_id in ids:
            idx = np.where(ids == common_id)[0][0]
            poses1.append(all_poses1[i][0][idx].flatten())
            corner_points.append(all_corners1[i][idx].reshape(-1, 2))

    for i, ids in enumerate(all_ids2):
        if common_id in ids:
            idx = np.where(ids == common_id)[0][0]
            poses2.append(all_poses2[i][0][idx].flatten())

    corner_points = np.mean(corner_points, axis=0)

    best_pose_index = find_best_pose(poses1, corner_points)

    best_rvec1, best_tvec1 = all_poses1[best_pose_index]

    R, T = relative_camera_pose(best_rvec1, best_tvec1[0], all_poses2[best_pose_index][0], all_poses2[best_pose_index][1][0])

    save_rt_matrices(R, T)

    print(f"Best pose for marker {common_id} between camera 1 and camera 2:")
    print(f"Rotation: \n{R}")
    print(f"Translation: \n{T}")