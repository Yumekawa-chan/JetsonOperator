import open3d as o3d
import cv2
import numpy as np

def get_transformation_matrix(R, Tr):
    
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = Tr.T

    print("transform_mtx:\n",transformation_matrix)
    
    return transformation_matrix

def create_point_cloud_from_mask(depth_image_path, color_image_path, intrinsics, mask_image_path):
    print("depth_image_path:", depth_image_path)
    print("color_image_path:", color_image_path)
    print("mask_image_path:", mask_image_path)

    depth_image = np.load(depth_image_path).astype(np.float32)
    color_image = cv2.imread(color_image_path, cv2.IMREAD_COLOR)
    mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)

    fx, fy, cx, cy, *_ = intrinsics

    rows, cols = depth_image.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    valid = (depth_image > 0) & (mask_image == 255)
    z = np.where(valid, depth_image / 1000.0, 0)
    x = np.where(valid, z * (c - cx) / fx, 0)
    y = np.where(valid, z * (r - cy) / fy, 0)

    points = np.dstack((x, y, z))
    colors = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB) / 255.0
    points = points[valid]
    colors = colors[valid]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd
