import numpy as np
import open3d as o3d
from glob import glob
from utils.img2pcd import create_point_cloud_from_mask, get_transformation_matrix
from utils.pcd_process import process_outliers
from utils.mask_maker import make_mask
import os

def load_intrinsics_and_image_paths(camera_id):
    intrinsics_path = f'matrix/depth_intrinsics_{camera_id}.npy'
    depth_image_paths = sorted(glob(f'image/depth_*_{camera_id}.npy'))
    color_image_paths = sorted(glob(f'image/color_*_{camera_id}.png'))

    make_mask(os.path.abspath(color_image_paths[0]), camera_id)
    
    mask_image_paths = glob(f'./mask/mask_{camera_id}.png')

    return (os.path.abspath(depth_image_paths[0]), 
            os.path.abspath(color_image_paths[0]),
            np.load(os.path.abspath(intrinsics_path)), 
            os.path.abspath(mask_image_paths[0]))

def load_transformation_matrices(rotation_id, translation_id):
    R = np.load(f'matrix/R_{rotation_id}.npy')
    T = np.load(f'matrix/T_{translation_id}.npy')
    return get_transformation_matrix(R, T)

def apply_transformation(pcd_list, transformation_matrix):
    for pcd in pcd_list:
        pcd.transform(transformation_matrix)

def main():
    intrinsics_and_paths = {i: load_intrinsics_and_image_paths(i) for i in range(1, 5)}
    print(intrinsics_and_paths[2])

    pcds = {}
    for i in range(1, 5):
        pcds[i] = create_point_cloud_from_mask(*intrinsics_and_paths[i])
        print(f"Camera {i}: Point cloud has {len(pcds[i].points)} points.")
        print("-----------------------------------")

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
