'''
conda activate view-of-delft-env
python -m scripts.static_voxel_map
'''

import pyvista as pv
from voxel_world.frame_loader import get_pointcloud, get_origin
from voxel_world.pc_processing import generate_freespace_pointcloud, generate_voxel_array_dense, voxel_matrix_to_coords, get_freespace_matrix
from voxel_world.plotter_functions import draw_domain, draw_voxels, draw_coordinate_axes, draw_pointcloud

#Define Frame Range
START_FRAME = 10        #first frame in the sequence
END_FRAME = 530         #last frame in the sequence

#Define Map-Type
REFERENCE_FRAME = "map" #options: "lidar", "map"
SENSOR = "radar"        #options: "lidar", "radar"

#Voxel-Parameters
X_RANGE = (-1250, -1200)
Y_RANGE = (1500, 1550)
Z_RANGE = (-5, 5)
CUBE_SIZE = 0.5         #Voxel-Map resolution
POINTS_PER_RAY = 20     #For freespace modelling
FREESPACE_TRESHOLD = 0
OCCUPIED_TRESHOLD = 0

#Plotting Options
REFRESH = False
DRAW_POINTCLOUD = True
DRAW_OCCUPIED = True
DRAW_FREE = False
DRAW_MAP_FRAME = False



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

    # 2) Load point cloud for this frame
    frame_str = f"{int(frame_number):05d}"
    
    pc = get_pointcloud(frame_str, reference_frame=REFERENCE_FRAME, sensor=SENSOR)
    sensor_origin = get_origin(SENSOR, REFERENCE_FRAME, frame_str)

    # 3) Generate & draw voxels
    occupied_voxel_matrix, translation = generate_voxel_array_dense(pc, cube_size, x_range, y_range, z_range)
    voxel_coords = voxel_matrix_to_coords(occupied_voxel_matrix, cube_size, translation, threshold=OCCUPIED_TRESHOLD)
    if DRAW_OCCUPIED:
        draw_voxels(plotter, voxel_coords, cube_size=cube_size)

    # 4) Draw pointcloud
    if DRAW_POINTCLOUD:
        draw_pointcloud(plotter, pc[:, :3], color="red", point_size=2)

    # 5) Draw Freespace
    try:
        freespace_pc = generate_freespace_pointcloud(voxel_coords, sensor_origin, POINTS_PER_RAY)
        free_voxel_matrix, translation = generate_voxel_array_dense(freespace_pc, cube_size, x_range, y_range, z_range)
        free_voxels = get_freespace_matrix(free_voxel_matrix, occupied_voxel_matrix)
        freespace_voxel_coords = voxel_matrix_to_coords(free_voxels, cube_size, translation, threshold=FREESPACE_TRESHOLD)
        if DRAW_FREE:
            draw_voxels(plotter, freespace_voxel_coords, cube_size=cube_size, color="grey")
    except Exception as e:
        print(f"An error occurred modelling dynamic space: {e}")
    

    # 6) (Optional) coordinate axes if you want them every frame
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

