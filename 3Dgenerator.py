import numpy as np
import cv2
import open3d as o3d

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

depth_intrinsics_1 = np.load('depth_intrinsics_1.npy')
depth_image_path_1 = 'image/depth_20231211_102200_1.png'
color_image_path_1 = 'image/color_20231211_102200_1.png'

depth_intrinsics_2 = np.load('depth_intrinsics_2.npy')
depth_image_path_2 = 'image/depth_20231211_102200_2.png'
color_image_path_2 = 'image/color_20231211_102200_2.png'

pcd_1 = create_point_cloud(depth_image_path_1, color_image_path_1, depth_intrinsics_1)
pcd_2 = create_point_cloud(depth_image_path_2, color_image_path_2, depth_intrinsics_2)

o3d.visualization.draw_geometries([pcd_1]+[pcd_2])


