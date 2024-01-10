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

def get_transformation_matrix(R, T):
    
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = T.T
    
    return transformation_matrix


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

def remove_outliers(pcd, nb_neighbors=50, std_ratio=2.0, radius=0.05, min_nb_points=10):
    pcd, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    pcd, ind = pcd.remove_radius_outlier(
        nb_points=min_nb_points, radius=radius)
    return pcd

def process_outliers(pcd_r, pcd_l):
    print("Processing the reference point cloud...")
    pcd_r_processed = remove_outliers(pcd_r)
    print("Processing the point cloud to transform...")
    pcd_l_processed = remove_outliers(pcd_l)
    return pcd_r_processed, pcd_l_processed

depth_intrinsics_1 = np.load('matrix/depth_intrinsics_1.npy')
depth_image_path_1 = glob('image/depth_*_1.png')[0]
color_image_path_1 = glob('image/color_*_1.png')[0]

depth_intrinsics_2 = np.load('matrix/depth_intrinsics_2.npy')
depth_image_path_2 = glob('image/depth_*_2.png')[0]
color_image_path_2 = glob('image/color_*_2.png')[0]

pcd_1 = create_point_cloud(depth_image_path_1, color_image_path_1, depth_intrinsics_1)
pcd_2 = create_point_cloud(depth_image_path_2, color_image_path_2, depth_intrinsics_2)


R = np.load('matrix/R.npy')
T = np.load('matrix/T.npy')

transformation_matrix = get_transformation_matrix(R, T)

pcd_1.transform(transformation_matrix)

# pcd_1,pcd_2 = process_outliers(pcd_1,pcd_2)

pcd_1,pcd_2 = execute_icp(pcd_1,pcd_2)

o3d.visualization.draw_geometries([pcd_1]+[pcd_2])


