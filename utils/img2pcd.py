import open3d as o3d
import cv2
import numpy as np

def get_transformation_matrix(R, Tr):
    
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = Tr.T

    print("transform_mtx:\n",transformation_matrix)
    
    return transformation_matrix

def create_point_cloud(depth_image_path, color_image_path, intrinsics):
    depth_image = np.load(depth_image_path)
    depth_image = depth_image.astype(np.uint16)
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