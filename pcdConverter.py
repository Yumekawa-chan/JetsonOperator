import numpy as np
import open3d as o3d
from glob import glob
from utils.img2pcd import create_point_cloud, get_transformation_matrix
from utils.pcd_process import process_outliers

def load_intrinsics_and_image_paths(camera_id):
    intrinsics_path = f'matrix/depth_intrinsics_{camera_id}.npy'
    depth_image_path = sorted(glob(f'image/depth_*_{camera_id}.npy'))[0]
    color_image_path = sorted(glob(f'image/color_*_{camera_id}.png'))[0]
    return np.load(intrinsics_path), depth_image_path, color_image_path

def load_transformation_matrices(rotation_id, translation_id):
    R = np.load(f'matrix/R_{rotation_id}.npy')
    T = np.load(f'matrix/T_{translation_id}.npy')
    return get_transformation_matrix(R, T)

def apply_transformation(pcd_list, transformation_matrix):
    for pcd in pcd_list:
        pcd.transform(transformation_matrix)

def main():
    intrinsics_and_paths = {i: load_intrinsics_and_image_paths(i) for i in range(1, 5)}

    pcds = {i: create_point_cloud(*intrinsics_and_paths[i]) for i in range(1, 5)}

    transformation_matrices = {
        '4_3': load_transformation_matrices('4_3', '4_3'),
        '3_1': load_transformation_matrices('3_1', '3_1'),
        '1_2': load_transformation_matrices('1_2', '1_2'),
    }

    apply_transformation([pcds[4]], transformation_matrices['4_3'])
    apply_transformation([pcds[4], pcds[3]], transformation_matrices['3_1'])
    apply_transformation([pcds[4], pcds[3], pcds[1]], transformation_matrices['1_2'])

    combined_pcd = sum(pcds.values(), o3d.geometry.PointCloud())
    processed_pcd = process_outliers(combined_pcd)

    o3d.visualization.draw_geometries([processed_pcd])

if __name__ == '__main__':
    main()