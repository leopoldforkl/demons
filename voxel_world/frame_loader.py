from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader
from vod.frame import FrameTransformMatrix
from vod.frame import transform_pcl, homogeneous_coordinates, homogeneous_transformation
import numpy as np
from config import DATASET_PATH
from config import OUTPUT_DIR

# Set dataset and output paths
dataset_root = DATASET_PATH
output_dir = OUTPUT_DIR

# Initialize KITTI-like location configuration
kitti_locations = KittiLocations(
    root_dir=dataset_root,
    output_dir=output_dir,
    frame_set_path="",
    pred_dir="",
)

def get_pointcloud_lidar(frame_number, reference_frame="map"):
    """
    Loads and transforms a LiDAR point cloud into a specified reference frame.

    Args:
        frame_number (str): Frame number as a string (e.g. "0010").
        reference_frame (str): Desired reference frame. Options: "lidar", "camera", "map", "radar".

    Returns:
        np.ndarray: Transformed point cloud in the chosen reference frame.
    """
    # Load frame data (e.g., point clouds and sensor metadata)
    frame_data = FrameDataLoader(kitti_locations=kitti_locations, frame_number=frame_number)

    # Access predefined transformation matrices
    transforms = FrameTransformMatrix(frame_data)

    # Raw point cloud in the LiDAR coordinate frame
    point_cloud_lidar_frame = frame_data.lidar_data

    if reference_frame == "lidar":
        point_cloud = point_cloud_lidar_frame

    elif reference_frame == "camera":
        point_cloud = transform_pcl(point_cloud_lidar_frame, transforms.t_camera_lidar)

    elif reference_frame == "map":
        # Load transformation matrices only when needed
        odom2cam, map2cam, utm2cam = transforms.get_world_transform()
        point_cloud_camera_frame = transform_pcl(point_cloud_lidar_frame, transforms.t_camera_lidar)
        point_cloud = transform_pcl(point_cloud_camera_frame, map2cam)

    elif reference_frame == "radar":
        point_cloud_camera_frame = transform_pcl(point_cloud_lidar_frame, transforms.t_camera_lidar)
        point_cloud = transform_pcl(point_cloud_camera_frame, transforms.t_radar_camera)

    else:
        raise ValueError(f"Invalid reference_frame: '{reference_frame}'. Choose from 'lidar', 'camera', 'map', 'radar'.")

    return point_cloud

def get_pointcloud_radar(frame_number, reference_frame="map"):
    """
    Loads and transforms a radar point cloud into a specified reference frame.

    Args:
        frame_number (str): Frame number as a string (e.g. "0010").
        reference_frame (str): Desired reference frame. Options: "lidar", "camera", "map", "radar".

    Returns:
        np.ndarray: Transformed point cloud in the chosen reference frame.
    """
    # Load frame data (e.g., point clouds and sensor metadata)
    frame_data = FrameDataLoader(kitti_locations=kitti_locations, frame_number=frame_number)

    # Access predefined transformation matrices
    transforms = FrameTransformMatrix(frame_data)

    # Raw point cloud in the radar coordinate frame
    point_cloud_radar_frame = frame_data.radar_data

    if reference_frame == "radar":
        point_cloud = point_cloud_radar_frame

    elif reference_frame == "camera":
        point_cloud = transform_pcl(point_cloud_radar_frame, transforms.t_camera_radar)

    elif reference_frame == "map":
        # Load transformation matrices only when needed
        odom2cam, map2cam, utm2cam = transforms.get_world_transform()
        point_cloud_camera_frame = transform_pcl(point_cloud_radar_frame, transforms.t_camera_radar)
        point_cloud = transform_pcl(point_cloud_camera_frame, map2cam)

    elif reference_frame == "lidar":
        point_cloud_camera_frame = transform_pcl(point_cloud_radar_frame, transforms.t_camera_radar)
        point_cloud = transform_pcl(point_cloud_camera_frame, transforms.t_lidar_camera)

    else:
        raise ValueError(f"Invalid reference_frame: '{reference_frame}'. Choose from 'lidar', 'camera', 'map', 'radar'.")

    return point_cloud

def get_pointcloud(frame_number, reference_frame="map", sensor="lidar"):
    
    if sensor =="lidar":
        point_cloud = get_pointcloud_lidar(frame_number, reference_frame)
    elif sensor =="radar":
        point_cloud = get_pointcloud_radar(frame_number, reference_frame)
    else:
        raise ValueError(f"Invalid sensor: '{reference_frame}'. Choose from 'lidar', 'radar'.")

    return point_cloud

def get_origin(origin, target_frame, frame_number):
    '''
    computes the origin of some frame (origin = "camera", "lidar", "radar", "map") in a target_frame (target_frame = "camera", "lidar", "radar", "map")
    at a given frame. the output is a 3x1 numpy array of the origin coordinates.
    '''
    zero_vector = np.zeros((1, 3))
    vector = homogeneous_coordinates(zero_vector)

    frame_data = FrameDataLoader(kitti_locations=kitti_locations, frame_number=frame_number)
    transforms = FrameTransformMatrix(frame_data)
    odom_to_camera, map_to_camera, utm_to_camera = transforms.get_world_transform()

    #first transform the origin to the camera frame
    if origin == "radar":
        vector = homogeneous_transformation(vector, transforms.t_camera_radar)

    elif origin == "camera":
        vector = vector

    elif origin == "map":
        vector = homogeneous_transformation(vector, np.linalg.inv(map_to_camera))

    elif origin == "lidar":
        vector = homogeneous_transformation(vector, transforms.t_camera_lidar)

    else:
        raise ValueError(f"Invalid reference_frame: '{origin}'. Choose from 'lidar', 'camera', 'map', 'radar'.")
    
    #now transform from the camera frame to the desired frame
    if target_frame == "radar":
        vector = homogeneous_transformation(vector, transforms.t_radar_camera)

    elif target_frame == "camera":
        vector = vector

    elif target_frame == "map":
        vector = homogeneous_transformation(vector, map_to_camera)

    elif target_frame == "lidar":
        vector = homogeneous_transformation(vector, transforms.t_lidar_camera)

    else:
        raise ValueError(f"Invalid reference_frame: '{origin}'. Choose from 'lidar', 'camera', 'map', 'radar'.")

    origin_coordinates = vector[0, :3]
    return origin_coordinates

def get_radar_velocity_vectors(pc_radar, compensated_radial_velocity):
    radial_unit_vectors = pc_radar / np.linalg.norm(pc_radar, axis=1, keepdims=True)
    velocity_vectors = compensated_radial_velocity[:, None] * radial_unit_vectors

    return velocity_vectors

def get_radar_velocities(frame_number, reference_frame="camera"):

    frame_data = FrameDataLoader(kitti_locations=kitti_locations, frame_number=frame_number)
    transforms = FrameTransformMatrix(frame_data)

    transform_matrix = transforms.t_camera_radar #transform_matrices['radar']

    compensated_radial_velocity = frame_data.radar_data[:, 5]
    radar_points_camera_frame = transform_pcl(points=frame_data.radar_data,
                                                  transform_matrix=transform_matrix)
    pc_radar = radar_points_camera_frame[:, 0:3]
    velocity_vectors = get_radar_velocity_vectors(pc_radar, compensated_radial_velocity) # velocity vectors in camera frame

    if reference_frame=="camera":
        velocity_vectors=velocity_vectors
    elif reference_frame=="map":
        odom2cam, map2cam, utm2cam = transforms.get_world_transform()
        # Set translation part to zero
        map2cam[0:3, 3] = 0
        velocity_vectors_homo=transform_pcl(velocity_vectors, map2cam)
        velocity_vectors = velocity_vectors_homo[:, :3]
    else:
        print("Warning: Invalid Reference Frame for radar velocity vectors!")
        velocity_vectors=None

    return velocity_vectors