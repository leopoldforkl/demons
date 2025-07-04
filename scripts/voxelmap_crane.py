'''
conda activate demons-env
python -m scripts.voxelmap_crane
'''

import pyvista as pv
from voxel_world.pc_processing import generate_freespace_pointcloud, generate_voxel_array_dense, voxel_matrix_to_coords, get_freespace_matrix
from voxel_world.plotter_functions import draw_domain, draw_voxels, draw_coordinate_axes, draw_pointcloud

import numpy as np
import json
import os

from config import ROS_DATASET_PATH


#Define Frame Range
START_FRAME = 0        #first frame in the sequence
END_FRAME = 100         #last frame in the sequence

#Voxel-Parameters
X_RANGE = (-50, 50)
Y_RANGE = (-50, 50)
Z_RANGE = (-50, 50)
CUBE_SIZE = 1.0         #Voxel-Map resolution
POINTS_PER_RAY = 20     #For freespace modelling
FREESPACE_TRESHOLD = 0
OCCUPIED_TRESHOLD = 0

#Plotting Options
REFRESH = False
DRAW_POINTCLOUD = True
DRAW_FREESPACE_POINTCLOUD = False
DRAW_OCCUPIED = False
DRAW_FREE = False
DRAW_MAP_FRAME = False

# Configuration
#ros_bag = 'ros2bag_20250331_092043'
ros_bag = 'ros2bag_20250331_100045'

meta_data_path = f'{ROS_DATASET_PATH}/{ros_bag}/metadata.json'
bag_path = f'{ROS_DATASET_PATH}/{ros_bag}'

# Load metadata
with open(meta_data_path, 'r') as f:
    metadata = json.load(f)

def load_frame(sensor_type, frame_index=0):
    """Load a specific frame's data with proper validation"""
    frame_info = metadata[sensor_type][frame_index]
    file_path = os.path.join(bag_path, frame_info['filename'])
    
    try:
        with open(file_path, 'rb') as f:
            # Read header
            point_count = np.fromfile(f, dtype='<u4', count=1)[0]
            has_velocity = np.fromfile(f, dtype='u1', count=1)[0]
            
            # Read points
            points = np.fromfile(f, dtype='<f4', count=point_count*3)
            points = points.reshape(-1, 3)
            
            if sensor_type == 'radar' and has_velocity:
                velocities = np.fromfile(f, dtype='<f4', count=point_count)
                return points, velocities.flatten(), frame_info  # Changed to 1D array
            else:
                return points, None, frame_info
                
    except Exception as e:
        print(f"Error loading {sensor_type} frame {frame_index}: {str(e)}")
        return np.zeros((0, 3)), None, frame_info


def get_pointcloud(frame, sensor):
    if sensor == 'radar':
        radar_points, radar_velocities, radar_info = load_frame('radar', frame)
        return radar_points
    elif sensor =='lidar':
        lidar_points, _, lidar_info = load_frame('lidar', frame)
        return lidar_points
    print("Invalid Sensor!")
    return None

def get_origin():
    origin = np.zeros((3, 1))
    return origin

def plot_scene(refresh, frame_number,
               x_range, y_range, z_range,
               cube_size,
               plotter):
    """
    Clear everything, then
      - draw the static domain
      - load & draw the voxels for this frame
      - load & draw the pointcloud for this frame
    """
    if refresh:
        # Clear all actors (domain, voxels, pointcloud, axes, etc)
        plotter.clear_actors()
        # 1) Static domain
        draw_domain(plotter, x_range, y_range, z_range)

    # 1) Load point cloud for this frame
    pc_radar = get_pointcloud(frame_number, 'radar')
    pc_lidar = get_pointcloud(frame_number, 'lidar')

    # 2) Draw pointcloud

    draw_pointcloud(plotter, pc_radar[:, :3], color="red", point_size=2)
    draw_pointcloud(plotter, pc_lidar[:, :3], color="blue", point_size=2)

    # 3) Generate & draw voxels
    sensor_origin = get_origin()
    occupied_voxel_matrix, translation = generate_voxel_array_dense(pc_lidar, cube_size, x_range, y_range, z_range)
    voxel_coords = voxel_matrix_to_coords(occupied_voxel_matrix, cube_size, translation, threshold=OCCUPIED_TRESHOLD)
    if DRAW_OCCUPIED:
        draw_voxels(plotter, voxel_coords, cube_size=cube_size)

    plotter.render()

def main():
    
    #Voxel-Parameters
    x_range = X_RANGE
    y_range = Y_RANGE
    z_range = Z_RANGE

    # Create plotter
    plotter = pv.Plotter()
    plotter.set_background("white")

    # Initial draw
    if not REFRESH:
        draw_domain(plotter, x_range, y_range, z_range)
    plot_scene(REFRESH, START_FRAME, x_range, y_range, z_range, CUBE_SIZE, plotter)

    # Slider callback just calls plot_scene again
    plotter.add_slider_widget(
        callback=lambda val: plot_scene(REFRESH, int(val), x_range, y_range, z_range, CUBE_SIZE, plotter),
        rng=[START_FRAME, END_FRAME],
        value=START_FRAME,
        title="Frame",
        fmt="%0.0f",
        color="black",
        title_color="black",
        interaction_event="always"
    )

    plotter.show()


if __name__ == "__main__":
    main()

