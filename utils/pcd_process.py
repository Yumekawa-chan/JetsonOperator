import open3d as o3d

def execute_icp(source, target):
    print("Apply initial alignment")
    trans_init = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance=0.3,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    ).transformation
    print(trans_init)

    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance=0.3,
        init=trans_init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    transformation_icp = reg_p2p.transformation
    print(transformation_icp)

    source.transform(transformation_icp)
    return source, target

def remove_outliers(pcd, nb_neighbors=32, std_ratio=2.0, radius=0.05, min_nb_points=20):
    pcd_statistical_filtered, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    pcd_radius_filtered, _ = pcd_statistical_filtered.remove_radius_outlier(
        nb_points=min_nb_points, radius=radius)

    return pcd_radius_filtered


def process_outliers(pcd):
    pcd_processed = remove_outliers(pcd)
    return pcd_processed