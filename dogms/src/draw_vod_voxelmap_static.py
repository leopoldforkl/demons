

import pyvista as pv
from dogms.src.vod.frame_loader import get_pointcloud, get_origin
from dogms.src.pc_processing import generate_freespace_pointcloud, generate_voxel_array_dense, voxel_matrix_to_coords, get_freespace_matrix
from dogms.src.plotter_functions import draw_domain, draw_voxels, draw_coordinate_axes, draw_pointcloud

import pyvista as pv
import json
from dogms.src.vod.frame_loader import get_pointcloud, get_origin
from dogms.src.pc_processing import generate_freespace_pointcloud, generate_voxel_array_dense, voxel_matrix_to_coords, get_freespace_matrix
from dogms.src.plotter_functions import draw_domain, draw_voxels, draw_coordinate_axes, draw_pointcloud

def plot_scene(refresh, frame_number, config, plotter):
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
        draw_domain(plotter, config["x_range"], config["y_range"], config["z_range"])

    # 2) Load point cloud for this frame
    frame_str = f"{int(frame_number):05d}"
    
    pc = get_pointcloud(frame_str, reference_frame=config["reference_frame"], sensor=config["sensor"])
    sensor_origin = get_origin(config["sensor"], config["reference_frame"], frame_str)

    # 3) Generate & draw voxels
    occupied_voxel_matrix, translation = generate_voxel_array_dense(
        pc, 
        config["cube_size"], 
        config["x_range"], 
        config["y_range"], 
        config["z_range"]
    )
    voxel_coords = voxel_matrix_to_coords(
        occupied_voxel_matrix, 
        config["cube_size"], 
        translation, 
        threshold=config["occupied_threshold"]
    )
    if config["draw_occupied"]:
        draw_voxels(plotter, voxel_coords, cube_size=config["cube_size"])

    # 4) Draw pointcloud
    if config["draw_pointcloud"]:
        draw_pointcloud(plotter, pc[:, :3], color="red", point_size=2)

    # 5) Draw Freespace
    try:
        freespace_pc = generate_freespace_pointcloud(
            voxel_coords, 
            sensor_origin, 
            config["points_per_ray"]
        )
        free_voxel_matrix, translation = generate_voxel_array_dense(
            freespace_pc, 
            config["cube_size"], 
            config["x_range"], 
            config["y_range"], 
            config["z_range"]
        )
        free_voxels = get_freespace_matrix(free_voxel_matrix, occupied_voxel_matrix)
        freespace_voxel_coords = voxel_matrix_to_coords(
            free_voxels, 
            config["cube_size"], 
            translation, 
            threshold=config["freespace_threshold"]
        )
        if config["draw_free"]:
            draw_voxels(plotter, freespace_voxel_coords, cube_size=config["cube_size"], color="grey")
        if config["draw_freespace_pointcloud"]:
            draw_pointcloud(plotter, freespace_pc[:, :3], color="blue", point_size=2)
    except Exception as e:
        print(f"An error occurred modelling dynamic space: {e}")
    
    # 6) (Optional) coordinate axes if you want them every frame
    if config["draw_map_frame"]:
        draw_coordinate_axes(plotter)

    plotter.render()

def run(config):
    
    print("Running with config:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Create plotter
    plotter = pv.Plotter()
    plotter.set_background("white")

    # Initial draw
    if not config["refresh"]:
        draw_domain(plotter, config["x_range"], config["y_range"], config["z_range"])
    
    # Slider callback
    plotter.add_slider_widget(
        callback=lambda val: plot_scene(
            config["refresh"], 
            int(val), 
            config,
            plotter
        ),
        rng=[config["start_frame"], config["end_frame"]],
        value=config["start_frame"],
        title="Frame",
        fmt="%0.0f",
        color="black",
        title_color="black",
        interaction_event="always"
    )

    plotter.show()


