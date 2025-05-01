import pyvista as pv
import numpy as np


def draw_domain(plotter, x_range=(-2.5, 2.5), y_range=(-2.5, 2.5), z_range=(-2.5, 2.5),
                style="wireframe", color="black", line_width=2, opacity=0.5):
    """
    Draws a domain cube (wireframe) representing the empty space.
    
    Parameters:
        plotter (pv.Plotter): The PyVista plotter instance.
        x_range (tuple): The x-axis range (x_min, x_max).
        y_range (tuple): The y-axis range (y_min, y_max).
        z_range (tuple): The z-axis range (z_min, z_max).
        style (str): The style of the mesh (e.g., "wireframe").
        color (str): The color of the mesh.
        line_width (int): The width of the lines for wireframe style.
        opacity (float): The opacity of the cube.
    """
    # Calculate center coordinates
    center = (
        (x_range[0] + x_range[1]) / 2,
        (y_range[0] + y_range[1]) / 2,
        (z_range[0] + z_range[1]) / 2
    )
    
    # Calculate dimensions
    dims = (
        x_range[1] - x_range[0],
        y_range[1] - y_range[0],
        z_range[1] - z_range[0]
    )
    
    domain = pv.Cube(center=center, 
                     x_length=dims[0], 
                     y_length=dims[1], 
                     z_length=dims[2])
    plotter.add_mesh(domain, style=style, color=color, line_width=line_width, opacity=opacity)


def draw_voxels(plotter, voxel_coords, cube_size=1, color="blue"):
    """
    Ultra-fast voxel rendering using point glyphs.
    """
    if len(voxel_coords) == 0:
        return

    voxel_coords = np.asarray(voxel_coords)
    
    # Create points at voxel centers
    points = pv.PolyData(voxel_coords)
    
    # Glyph the points with a cube
    cube = pv.Cube(x_length=cube_size, y_length=cube_size, z_length=cube_size)
    voxels = points.glyph(geom=cube, orient=False, scale=False)
    
    plotter.add_mesh(voxels, color=color)


def draw_coordinate_axes(plotter, scale_factor = 1, arrow_tip_length=0.2, arrow_tip_radius=0.05, arrow_shaft_radius=0.02):
    """
    Draws coordinate axes at the origin using arrows with
    red for x, green for y, and blue for z.
    
    Parameters:
        plotter (pv.Plotter): The PyVista plotter instance.
        arrow_tip_length (float): Length of the arrow tip.
        arrow_tip_radius (float): Radius of the arrow tip.
        arrow_shaft_radius (float): Radius of the arrow shaft.
    """
    # X-axis (red)
    arrow_x = pv.Arrow(
        start=(0, 0, 0),
        direction=(1, 0, 0),
        tip_length=arrow_tip_length,
        tip_radius=arrow_tip_radius,
        shaft_radius=arrow_shaft_radius,
        scale=scale_factor,
    )
    plotter.add_mesh(arrow_x, color="red")
    
    # Y-axis (green)
    arrow_y = pv.Arrow(
        start=(0, 0, 0),
        direction=(0, 1, 0),
        tip_length=arrow_tip_length,
        tip_radius=arrow_tip_radius,
        shaft_radius=arrow_shaft_radius,
        scale=scale_factor,
    )
    plotter.add_mesh(arrow_y, color="green")
    
    # Z-axis (blue)
    arrow_z = pv.Arrow(
        start=(0, 0, 0),
        direction=(0, 0, 1),
        tip_length=arrow_tip_length,
        tip_radius=arrow_tip_radius,
        shaft_radius=arrow_shaft_radius,
        scale=scale_factor,
    )
    plotter.add_mesh(arrow_z, color="blue")

def draw_pointcloud(plotter, points, color="red", point_size=5):
    """
    Draws a point cloud from externally generated points.
    
    Parameters:
        plotter (pv.Plotter): The PyVista plotter instance.
        points (np.ndarray or list): A collection of points, each defined as (x, y, z).
        color (str): The color of the points.
        point_size (int): Size of the points.
    """
    plotter.add_points(points, color=color, point_size=point_size)