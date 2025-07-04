'''
conda activate demons-env
python -m scripts.for_paper

This script uses radar and lidar data to detect static, dynamic and free voxels in a scene
'''

import pyvista as pv
from voxel_world.frame_loader import get_pointcloud, get_origin, get_radar_velocities
from voxel_world.pc_processing import generate_freespace_pointcloud, generate_voxel_array_dense, voxel_matrix_to_coords, generate_velocity_voxels
from voxel_world.plotter_functions import draw_domain, draw_voxels

import numpy as np
from scipy.ndimage import gaussian_filter

#Define Frame Range
START_FRAME = 65        #first frame in the sequence
END_FRAME = 130         #last frame in the sequence

#Voxel-Parameters
X_RANGE = (-1240, -1210)
Y_RANGE = (1525, 1550)
Z_RANGE = (-5, 5)
CUBE_SIZE = 0.1         #Voxel-Map resolution
POINTS_PER_RAY = 50     #For freespace modelling
FREESPACE_TRESHOLD = 0
OCCUPIED_TRESHOLD = 0


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
    accumulated_dynamic = np.zeros((nx, ny, nz), dtype=int)


    for frame_number in range(start_frame, end_frame+1):
        frame_str = f"{int(frame_number):05d}"
    
        # 1) Load point and vector clouds for this frame
        pc_radar = get_pointcloud(frame_str, reference_frame="map", sensor="radar")
        pc_lidar = get_pointcloud(frame_str, reference_frame="map", sensor="lidar")
        sensor_origin_lidar = get_origin("lidar", "map", frame_str)
        velocity_vectors = get_radar_velocities(frame_str, "map", v_min=0.0, v_max=5.0)

        

        # 2) Operations for freespace-modelling
        #Radar:
        occupied_voxel_matrix_r, translation_r = generate_voxel_array_dense(pc_radar, cube_size, x_range, y_range, z_range)
        voxel_coords_4_freespace = voxel_matrix_to_coords(occupied_voxel_matrix_r, cube_size, translation_r, threshold=0)
        #Lidar:
        occupied_voxel_matrix_l, translation_l = generate_voxel_array_dense(pc_lidar, cube_size, x_range, y_range, z_range)
        voxel_coords_4_freespace = voxel_matrix_to_coords(occupied_voxel_matrix_l, cube_size, translation_l, threshold=0)
        freespace_pc_lidar = generate_freespace_pointcloud(voxel_coords_4_freespace, sensor_origin_lidar, POINTS_PER_RAY)

        # 3) Operations in voxelspace
        velocity_matrix = generate_velocity_voxels(pc_radar, velocity_vectors, cube_size, x_range, y_range, z_range)
        scalar_velocity_matrix = np.linalg.norm(velocity_matrix, axis=-1)  # Shape (nx, ny, nz, 3) to Shape (nx, ny, nz)
        # Apply 3D Gaussian blur with sigma controlling the blur strength
        blurred_velocity_matrix = gaussian_filter(scalar_velocity_matrix, sigma=1.0)  # sigma can be a tuple (sx, sy, sz) for anisotropic blur

        occupied_voxel_matrix = occupied_voxel_matrix_l                                                                         #use lidar, can be changed to radar
        free_voxel_matrix, translation = generate_voxel_array_dense(freespace_pc_lidar, cube_size, x_range, y_range, z_range)   #use lidar, can be changed to radar

        #accumulate
        normalized_occupied = (occupied_voxel_matrix > OCCUPIED_TRESHOLD).astype(int)
        normalized_free = (free_voxel_matrix > FREESPACE_TRESHOLD).astype(int)
        normalized_dynamic = (blurred_velocity_matrix > 0.02).astype(int)

        accumulated_occupied += normalized_occupied
        accumulated_free += normalized_free
        accumulated_dynamic += normalized_dynamic


    denom = accumulated_occupied + accumulated_free
    p_occupied = np.divide(accumulated_occupied, denom, out=np.zeros_like(accumulated_occupied, dtype=float), where=denom>=5)
    occupied_static = (p_occupied > 0.1).astype(int)
    occupied_dynamic = ((p_occupied > 0.0) & (p_occupied <= 0.1)).astype(int)

    # Drawing Operations
    draw_voxels(plotter, voxel_matrix_to_coords((accumulated_dynamic * occupied_dynamic), cube_size, translation, threshold=0.0), cube_size=cube_size, color=[1.0, 0.75, 0.0]) #orange
    draw_voxels(plotter, voxel_matrix_to_coords(occupied_static, cube_size, translation, threshold=0), cube_size=cube_size) #default blue



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

