'''
conda activate demons-env
python -m scripts.hybrid_voxel_map

This script uses radar and lidar data to detect static, dynamic and free voxels in a scene
'''

import pyvista as pv
from voxel_world.frame_loader import get_pointcloud, get_origin, get_radar_velocities
from voxel_world.pc_processing import generate_freespace_pointcloud, generate_voxel_array_dense, voxel_matrix_to_coords, get_freespace_matrix, generate_velocity_voxels, velocity_voxel_matrix_to_coords
from voxel_world.plotter_functions import draw_domain, draw_voxels, draw_coordinate_axes, draw_pointcloud, draw_velocities

import numpy as np

#Define Frame Range
START_FRAME = 10        #first frame in the sequence
END_FRAME = 30         #last frame in the sequence

#Voxel-Parameters
X_RANGE = (-1250, -1200)
Y_RANGE = (1500, 1550)
Z_RANGE = (-5, 5)
CUBE_SIZE = 0.5         #Voxel-Map resolution
POINTS_PER_RAY = 20     #For freespace modelling
FREESPACE_TRESHOLD = 0
OCCUPIED_TRESHOLD = 0

#Plotting Options
DRAW_MAP_FRAME = False


def plot_scene(start_frame, end_frame,
               x_range, y_range, z_range,
               cube_size,
               plotter):
    """
    Clear everything, then
      - draw the static domain
      - load & draw the voxels for this frame
      - load & draw the pointcloud for this frame
    """

    # 1) Static domain
    draw_domain(plotter, x_range, y_range, z_range)

    # compute number of voxels along each axis
    nx = int(np.floor((x_range[1] - x_range[0]) / cube_size))
    ny = int(np.floor((y_range[1] - y_range[0]) / cube_size))
    nz = int(np.floor((z_range[1] - z_range[0]) / cube_size))
    
    accumulated_occupied = np.zeros((nx, ny, nz), dtype=int)
    accumulated_free = np.zeros((nx, ny, nz), dtype=int)


    for frame_number in range(start_frame, end_frame+1):
        frame_str = f"{int(frame_number):05d}"
    
        # 1) Load point and vector clouds for this frame
        pc_radar = get_pointcloud(frame_str, reference_frame="map", sensor="radar")
        pc_lidar = get_pointcloud(frame_str, reference_frame="map", sensor="lidar")
        sensor_origin_radar = get_origin("radar", "map", frame_str)
        sensor_origin_lidar = get_origin("lidar", "map", frame_str)
        velocity_vectors = get_radar_velocities(frame_str, "map", v_min=0.0, v_max=5.0)

        

        # 2) Operations for freespace-modelling
        #Radar:
        occupied_voxel_matrix_r, translation_r = generate_voxel_array_dense(pc_radar, cube_size, x_range, y_range, z_range)
        voxel_coords_4_freespace = voxel_matrix_to_coords(occupied_voxel_matrix_r, cube_size, translation_r, threshold=0)
        freespace_pc_radar = generate_freespace_pointcloud(voxel_coords_4_freespace, sensor_origin_radar, POINTS_PER_RAY)
        #Lidar:
        occupied_voxel_matrix_l, translation_l = generate_voxel_array_dense(pc_lidar, cube_size, x_range, y_range, z_range)
        voxel_coords_4_freespace = voxel_matrix_to_coords(occupied_voxel_matrix_l, cube_size, translation_l, threshold=0)
        freespace_pc_lidar = generate_freespace_pointcloud(voxel_coords_4_freespace, sensor_origin_lidar, POINTS_PER_RAY)

        # 3) Operations in voxelspace
        velocity_matrix = generate_velocity_voxels(pc_radar, velocity_vectors, cube_size, x_range, y_range, z_range)

        occupied_voxel_matrix = occupied_voxel_matrix_l                                                                         #use lidar, can be changed to radar
        free_voxel_matrix, translation = generate_voxel_array_dense(freespace_pc_lidar, cube_size, x_range, y_range, z_range)   #use lidar, can be changed to radar

        draw_voxels(plotter, voxel_matrix_to_coords(occupied_voxel_matrix_r, cube_size, translation, threshold=0), cube_size=cube_size, color=[1.0, 0.75, 0.0]) #orange
        final_voxelcloud_r, voxel_velocity_vectors = velocity_voxel_matrix_to_coords(occupied_voxel_matrix_r, velocity_matrix, cube_size, translation, threshold=0)
        draw_velocities(plotter, final_voxelcloud_r, voxel_velocity_vectors, scale_factor=2.0, color="green")

    '''
    ToDo:
    I want to plot all voxels as occupied which are occupied in over 90% of the frames with blue
    I want to plot all voxels that are currently occupied but not yet blue in orange
    I want to plot all orange voxels which contain a velocity vector (of some minimum size) in red
    '''

    # Drawing Operations
    draw_pointcloud(plotter, pc_lidar[:, :3], color="red", point_size=2)
    draw_voxels(plotter, voxel_matrix_to_coords(occupied_voxel_matrix_l, cube_size, translation, threshold=0), cube_size=cube_size) #default blue

    #(Optional) coordinate axes if you want them every frame
    if DRAW_MAP_FRAME:
        draw_coordinate_axes(plotter)

    plotter.render()


def main():
    
    #Voxel-Parameters
    x_range = X_RANGE
    y_range = Y_RANGE
    z_range = Z_RANGE

    # Create plotter
    plotter = pv.Plotter()
    plotter.set_background("white")
    print("plotting...")
    plot_scene(START_FRAME, END_FRAME, x_range, y_range, z_range, CUBE_SIZE, plotter)


    plotter.show()
    print("terminated")


if __name__ == "__main__":
    main()

