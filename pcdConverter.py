import numpy as np
import cv2
import open3d as o3d
from glob import glob

def create_point_cloud(depth_image_path, color_image_path, intrinsics):
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
    color_image = cv2.imread(color_image_path, cv2.IMREAD_COLOR)
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    fx, fy, cx, cy, *_ = intrinsics

    depth_o3d = o3d.geometry.Image(depth_image)
    color_o3d = o3d.geometry.Image(color_image)
    intrinsics_o3d = o3d.camera.PinholeCameraIntrinsic(
        depth_image.shape[1], depth_image.shape[0], fx, fy, cx, cy
    )

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, depth_scale=1000.0, depth_trunc=8.0, convert_rgb_to_intensity=False
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsics_o3d
    )

    return pcd

def get_transformation_matrix(R, Tr):
    
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = Tr.T

    print("transform_mtx:\n",transformation_matrix)
    
    return transformation_matrix

def all_icp(pcd1,pcd2,pcd3):
    pcd1,pcd2 = execute_icp(pcd1,pcd2)
    pcd_1_2 = pcd1 + pcd2
    pcd_1_2,pcd3 = execute_icp(pcd_1_2,pcd3)
    pcd_1_2_3 = pcd_1_2 + pcd3
    return pcd_1_2_3

def execute_icp(source, target):
    print("Apply initial alignment")
    trans_init = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance=0.3,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    ).transformation
    print(trans_init)

    print("Apply ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance=0.3,
        init=trans_init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    transformation_icp = reg_p2p.transformation
    print(transformation_icp)

    source.transform(transformation_icp)
    return source, target

def remove_outliers(pcd, nb_neighbors=100, std_ratio=2.0, radius=0.05, min_nb_points=5):
    pcd, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    pcd, ind = pcd.remove_radius_outlier(
        nb_points=min_nb_points, radius=radius)
    return pcd

def process_outliers(pcd_1, pcd_2,pcd_3):
    print("Processing the reference point cloud...")
    pcd_1_processed = remove_outliers(pcd_1)
    print("Processing the point cloud to transform...")
    pcd_2_processed = remove_outliers(pcd_2)
    print("Processing the point cloud to transform...")
    pcd_3_processed = remove_outliers(pcd_3)
    return pcd_1_processed, pcd_2_processed, pcd_3_processed

depth_intrinsics_1 = np.load('matrix/depth_intrinsics_eto.npy')
depth_image_path_1 = sorted(glob('image/depth_*_eto.png'))[0]
color_image_path_1 = sorted(glob('image/color_*_eto.png'))[0]

depth_intrinsics_2 = np.load('matrix/depth_intrinsics_1.npy')
depth_image_path_2 = sorted(glob('image/depth_*_1.png'))[0]
color_image_path_2 = sorted(glob('image/color_*_1.png'))[0]

depth_intrinsics_3 = np.load('matrix/depth_intrinsics_2.npy')
depth_image_path_3 = sorted(glob('image/depth_*_2.png'))[0]
color_image_path_3 = sorted(glob('image/color_*_2.png'))[0]


pcd_1 = create_point_cloud(depth_image_path_1, color_image_path_1, depth_intrinsics_1)
pcd_2 = create_point_cloud(depth_image_path_2, color_image_path_2, depth_intrinsics_2)
pcd_3 = create_point_cloud(depth_image_path_3, color_image_path_3, depth_intrinsics_3)

R_eto_1 = np.load('matrix/R_eto_1.npy') 
T_eto_1 = np.load('matrix/T_eto_1.npy')

R_1_2 = np.load('matrix/R_1_2.npy')
T_1_2 = np.load('matrix/T_1_2.npy')


transformation_matrix_1_2 = get_transformation_matrix(R_1_2, T_1_2)
transformation_matrix_eto_1 = get_transformation_matrix(R_eto_1, T_eto_1)

pcd_1.transform(transformation_matrix_eto_1)
pcd_1.transform(transformation_matrix_1_2)
pcd_2.transform(transformation_matrix_1_2)


pcd_1,pcd_2,pcd_3 = process_outliers(pcd_1,pcd_2,pcd_3)

# pcd_1_2_3 = all_icp(pcd_1,pcd_2,pcd_3)
pcd_1_2_3 = pcd_1 + pcd_2 + pcd_3
o3d.visualization.draw_geometries([pcd_1_2_3])

