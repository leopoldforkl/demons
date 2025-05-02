'''
conda activate demons-env
python -m scripts.dynamic_voxel_map
'''

import pyvista as pv
from voxel_world.frame_loader import get_pointcloud, get_origin
from voxel_world.pc_processing import generate_freespace_pointcloud, generate_voxel_array_dense, voxel_matrix_to_coords, get_freespace_matrix
from voxel_world.plotter_functions import draw_domain, draw_voxels, draw_coordinate_axes, draw_pointcloud

import numpy as np

#Define Frame Range
START_FRAME = 10        #first frame in the sequence
END_FRAME = 200         #last frame in the sequence

#Define Map-Type
REFERENCE_FRAME = "map" #options: "lidar", "map"
SENSOR = "lidar"        #options: "lidar", "radar"

#Voxel-Parameters
X_RANGE = (-1250, -1200)
Y_RANGE = (1500, 1550)
Z_RANGE = (-5, 5)
CUBE_SIZE = 0.5         #Voxel-Map resolution
POINTS_PER_RAY = 20     #For freespace modelling
FREESPACE_TRESHOLD = 0
OCCUPIED_TRESHOLD = 5

#Plotting Options
DRAW_POINTCLOUD = False
DRAW_OCCUPIED = True
DRAW_FREE = False
DRAW_DYNAMIC = True
DRAW_MAP_FRAME = False

#Define how many frames should be between frames determining dynamic cells
DELTA_FRAMES = 5    #needs to be positive



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


    for frame_number in range(start_frame, end_frame+1, DELTA_FRAMES):
        frame_str = f"{int(frame_number):05d}"
    
        # 2) Load point cloud for this frame
        pc = get_pointcloud(frame_str, reference_frame=REFERENCE_FRAME, sensor=SENSOR)
        sensor_origin = get_origin(SENSOR, REFERENCE_FRAME, frame_str)

        # 3) Generate occupied voxels
        occupied_voxel_matrix, translation = generate_voxel_array_dense(pc, cube_size, x_range, y_range, z_range)
        voxel_coords = voxel_matrix_to_coords(occupied_voxel_matrix, cube_size, translation, threshold=OCCUPIED_TRESHOLD)

        # 4) Draw pointcloud
        if DRAW_POINTCLOUD:
            draw_pointcloud(plotter, pc[:, :3], color="red", point_size=2)

        # 5) Generate Freespace
        try:
            freespace_pc = generate_freespace_pointcloud(voxel_coords, sensor_origin, POINTS_PER_RAY)
            free_voxel_matrix, translation = generate_voxel_array_dense(freespace_pc, cube_size, x_range, y_range, z_range)
            free_voxels = get_freespace_matrix(free_voxel_matrix, occupied_voxel_matrix)
        except Exception as e:
            print(f"An error occurred modelling free space: {e}")
        
        # 6) accumulate
        normalized_occupied = (occupied_voxel_matrix > OCCUPIED_TRESHOLD).astype(int)
        normalized_free = (free_voxels > FREESPACE_TRESHOLD).astype(int)
        accumulated_occupied += normalized_occupied
        accumulated_free += normalized_free
        

    # 7) calculate and draw:
    denom = accumulated_occupied + accumulated_free
    p_occupied = np.divide(accumulated_occupied, denom, out=np.zeros_like(accumulated_occupied, dtype=float), where=denom!=0)
    occupied = (p_occupied > 0.8).astype(int)
    dynamic = ((p_occupied > 0.0) & (p_occupied <= 0.8)).astype(int)
    p_free = np.divide(accumulated_free, denom, out=np.zeros_like(accumulated_free, dtype=float), where=denom!=0)
    free = (p_free == 1.0).astype(int)

    #dynamic should be additional freespace except its currently occupied
    current_dynamic = dynamic *  occupied_voxel_matrix

    occupied_coords = voxel_matrix_to_coords(occupied, cube_size, translation, threshold=0)
    free_coords = voxel_matrix_to_coords(free, cube_size, translation, threshold=0)
    dynamic_coords = voxel_matrix_to_coords(current_dynamic, cube_size, translation, threshold=0)
    
    if DRAW_OCCUPIED:
            draw_voxels(plotter, occupied_coords, cube_size=cube_size)
    if DRAW_FREE:
            draw_voxels(plotter, free_coords, cube_size=cube_size, color="grey")
    if DRAW_DYNAMIC:
            draw_voxels(plotter, dynamic_coords, cube_size=cube_size, color="green")

    # 8) (Optional) coordinate axes if you want them every frame
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

