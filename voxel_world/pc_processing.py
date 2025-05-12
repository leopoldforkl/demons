import numpy as np
import time

'''
Note: switch from numpy to torch or cupy if GPU available!
'''

def get_freespace_matrix(free_voxel_matrix, occupied_voxel_matrix):
    """
    Identify voxels that are free and not occupied.

    Inputs:
      • free_voxel_matrix : np.ndarray
      • occupied_voxel_matrix : np.ndarray
        two arrays of identical shape, representing occupied and free voxels.

    Output:
      • true_free_voxel_matrix : np.ndarray of same shape
    """
    # ensure all inputs share the same shape
    if not (free_voxel_matrix.shape == occupied_voxel_matrix.shape):
        raise ValueError("All input arrays must have the same shape.")

    # mask out “free” entries where the voxel was actually occupied
    true_free_voxel_matrix = free_voxel_matrix * (occupied_voxel_matrix == 0)
    return true_free_voxel_matrix

def generate_freespace_pointcloud(sensor_point_cloud, sensor_origin, points_per_ray):
    """
    Generates a point cloud modeling the free space between the sensor origin and each point in the measured point cloud.

    Args:
    - sensor_point_cloud (np.ndarray):  Nx3 array of 3D points captured by the sensor.
    - sensor_origin (np.ndarray):       1x3 array representing the sensor's position, e.g., np.array([x, y, z]).
    - points_per_ray (int):             Number of intermediate points to generate along each ray from the origin to each point in the point cloud.

    Returns:
    - np.ndarray: (S*N)x3 array representing the free space point cloud, where S is points_per_ray.
    """

    # Compute the vector from the sensor origin to each point in the point cloud
    points = sensor_point_cloud - sensor_origin  # Shape: (N, 3)
    # Generate S equally spaced scalar values between 0 and 1 to interpolate along each ray
    scales = np.linspace(0, 1, points_per_ray + 2)[1:-1]  # Shape: (S,)
    # Start timing the operation

    start_time = time.perf_counter()
    # Reshape scales to broadcast properly across the point cloud
    scales = scales[:, None, None]  # Shape: (S, 1, 1)
    # Multiply each point vector by each scale to generate intermediate points along the rays
    scaled_pts = points[None, ...] * scales  # Shape: (S, N, 3)
    # Reshape to a 2D array of shape (S*N, 3)
    result = scaled_pts.reshape(-1, 3)
    # Translate all points back into world coordinates by adding the sensor origin
    freespace_pc = result + sensor_origin  # Shape: (S*N, 3)
    # End timing and print duration
    end_time = time.perf_counter()

    print(f"Elapsed time for computing freespace pointcloud: {end_time - start_time:.6f} seconds")

    return freespace_pc


def generate_voxel_array_dense(point_cloud, cube_size, range_x, range_y, range_z):
    """
    Compute full voxel matrix for points in the point cloud, with timing metrics.
    
    Args:
        point_cloud: (N, 3) array of 3D points (x, y, z) in meters.
        cube_size:   Voxel edge length in meters (e.g., 0.1 for 10cm voxels).
        range_x:     Tuple (min_x, max_x) defining the x-axis bounds.
        range_y:     Tuple (min_y, max_y) defining the y-axis bounds.
        range_z:     Tuple (min_z, max_z) defining the z-axis bounds.
    
    Returns:
        voxel_matrix: Counter mapping (i,j,k) voxel indices → number of points
        translation:  3d-vector applied to original
    """
    # Initialize timers
    start_time = time.time()
    total_points = len(point_cloud)
    translation = np.array([range_x[0], range_y[0], range_z[0]]) # to make sure all points in the range are positive!

    # compute number of voxels along each axis
    nx = int(np.floor((range_x[1] - range_x[0]) / cube_size))
    ny = int(np.floor((range_y[1] - range_y[0]) / cube_size))
    nz = int(np.floor((range_z[1] - range_z[0]) / cube_size))
    voxel_matrix = np.zeros((nx, ny, nz), dtype=int)

    if total_points > 0:
        # --- Step 1: Extract XYZ coordinates ---
        t0 = time.time()
        cloud_3d = point_cloud[:, :3] - translation  # shape (N, 3)
        t1 = time.time()
        print(f"1. Extract XYZ and translate PC: {t1 - t0:.6f}s ({(t1 - t0)/total_points:.6f}s per point)")
        
        # --- Step 2: Convert points to voxel indices ---
        t0 = time.time()
        voxel_indices = np.floor(cloud_3d / cube_size).astype(int)
        t1 = time.time()
        print(f"2. Voxel index conversion: {t1 - t0:.6f}s ({(t1 - t0)/total_points:.6f}s per point)")

        # --- Step 3: 
        # convert to sparse matrix in range ---
        t0 = time.time()
        valid_mask = (
            (voxel_indices[:,0] >= 0) & (voxel_indices[:,0] < nx) &
            (voxel_indices[:,1] >= 0) & (voxel_indices[:,1] < ny) &
            (voxel_indices[:,2] >= 0) & (voxel_indices[:,2] < nz)
        )
        valid_voxels = voxel_indices[valid_mask]
        # for each point, increment its voxel
        np.add.at(voxel_matrix, (valid_voxels[:,0], valid_voxels[:,1], valid_voxels[:,2]), 1)

        t1 = time.time()
        print(f"3. Conversion to matrix: {t1 - t0:.6f}s ({(t1 - t0)/total_points:.6f}s per point)")
        

        
        # --- Total time ---
        total_time = time.time() - start_time
        print(f"\nTotal execution time: {total_time:.6f}s ({total_time/total_points:.6f}s per point)")
        occupied = np.count_nonzero(voxel_matrix)
        print(f"Occupied voxels: {occupied} / {nx*ny*nz}")
    else:
        print("Warning: Empty Pointcloud!")
        
    
    return voxel_matrix, translation

def voxel_matrix_to_coords(voxel_matrix, cube_size, translation, threshold=0):
    """
    Converts a 3D voxel matrix into a list of voxel center coordinates in 3D space.

    Parameters:
    - voxel_matrix (np.ndarray): 3D array representing voxels, where values > threshold are occupied.
    - cube_size (float): The size of each voxel cube.
    - translation (np.ndarray): A 3-element array to shift all coordinates by.
    - threshold (float, optional): Minimum value to consider a voxel as "filled". Default is 0.

    Returns:
    - np.ndarray: An array of 3D coordinates representing the centers of filled voxels.
    """

    # Find indices of voxels that are filled (greater than threshold)
    t0 = time.time()
    filled_indices = np.argwhere(voxel_matrix > threshold)
    t1 = time.time()
    #print(f"Conversiontime voxel_matrix -> voxel_coords: {t1 - t0:.6f}s")

    # Scale indices by cube size to get voxel positions in space
    scaled_coords = filled_indices * cube_size

    # Adjust to get the center of each voxel
    centered_coords = scaled_coords + cube_size / 2

    # Apply translation to place voxels in the correct world position
    voxel_coords = centered_coords + translation

    return voxel_coords

def generate_velocity_voxels(point_cloud, velocity_vectors, cube_size, range_x, range_y, range_z):
    """
    Compute occupancy and velocity voxel matrix for points in the point cloud. Prints timing metrics.
    
    Args:
        point_cloud: (N, 3) array of 3D points (x, y, z) in meters.
        velocity_vectors: (N, 3) array of velocity vectors (vx, vy, vz)
        cube_size:   Voxel edge length in meters (e.g., 0.1 for 10cm voxels).
        range_x:     Tuple (min_x, max_x) defining the x-axis bounds.
        range_y:     Tuple (min_y, max_y) defining the y-axis bounds.
        range_z:     Tuple (min_z, max_z) defining the z-axis bounds.
    
    Returns:
        voxel_matrix: (nx,ny,nz) array counting points per voxel
        velocity_matrix: (nx,ny,nz,3) array of accumulated velocity vectors
        avg_velocity_matrix: (nx,ny,nz,3) array of average velocity vectors (0 where no points)
        translation: 3d-vector applied to original
    """
    # Initialize timers
    start_time = time.time()
    total_points = len(point_cloud)
    translation = np.array([range_x[0], range_y[0], range_z[0]]) # to make sure all points in the range are positive!

    # compute number of voxels along each axis
    nx = int(np.floor((range_x[1] - range_x[0]) / cube_size))
    ny = int(np.floor((range_y[1] - range_y[0]) / cube_size))
    nz = int(np.floor((range_z[1] - range_z[0]) / cube_size))
    voxel_matrix = np.zeros((nx, ny, nz), dtype=int)
    velocity_matrix = np.zeros((nx, ny, nz, 3), dtype=float)
    avg_velocity_matrix = np.zeros((nx, ny, nz, 3), dtype=float)

    if total_points > 0:
        # --- Step 1: Extract XYZ coordinates ---
        cloud_3d = point_cloud[:, :3] - translation  # shape (N, 3)
        
        # --- Step 2: Convert points to voxel indices ---
        voxel_indices = np.floor(cloud_3d / cube_size).astype(int)

        # --- Step 3: 
        # convert to sparse matrix in range ---
        valid_mask = (
            (voxel_indices[:,0] >= 0) & (voxel_indices[:,0] < nx) &
            (voxel_indices[:,1] >= 0) & (voxel_indices[:,1] < ny) &
            (voxel_indices[:,2] >= 0) & (voxel_indices[:,2] < nz)
        )
        valid_voxels = voxel_indices[valid_mask]
        valid_velocities = velocity_vectors[valid_mask]
        # for each point, increment its voxel
        np.add.at(voxel_matrix, (valid_voxels[:,0], valid_voxels[:,1], valid_voxels[:,2]), 1)

         # Accumulate velocities per voxel (for each x,y,z component)
        for d in range(3):  # x,y,z components
            np.add.at(velocity_matrix[:,:,:,d], 
                     (valid_voxels[:,0], valid_voxels[:,1], valid_voxels[:,2]), 
                     valid_velocities[:,d])
            
        # Compute average velocities (avoid division by zero)
        # Method 1: Using np.divide with where (more explicit)
        counts = voxel_matrix[..., np.newaxis]  # Add extra dim for broadcasting
        np.divide(velocity_matrix, counts, out=avg_velocity_matrix, where=counts!=0) #Double Check!!

        # --- Total time ---
        total_time = time.time() - start_time
        print(f"\nTotal execution time velocity voxels: {total_time:.6f}s ({total_time/total_points:.6f}s per point)")
        occupied = np.count_nonzero(voxel_matrix)
        print(f"Occupied voxels: {occupied} / {nx*ny*nz}")
    else:
        print("Warning: Empty Pointcloud!")
        
    
    return avg_velocity_matrix


def velocity_voxel_matrix_to_coords(voxel_matrix, velocity_matrix, cube_size, translation, threshold=0):
    """
    Convert voxel matrix and velocity matrix back to world coordinates and velocity vectors.

    Parameters:
    - voxel_matrix (np.ndarray): 3D array representing voxels, where values > threshold are occupied.
    - velocity_matrix (np.ndarray): Array of shape (nx, ny, nz, 3) representing velocity vectors for each voxel
    - cube_size (float): The size of each voxel cube.
    - translation (np.ndarray): A 3-element array to shift all coordinates by.
    - threshold (float, optional): Minimum value to consider a voxel as "filled". Default is 0.

    Returns:
    - voxel_coords: (Nx3) array representing coordinates of occupied voxels
    - velocity_vecs: (Nx3) array representing velocity vectors for each occupied voxel
    """

    # Find indices of voxels that are filled (greater than threshold)
    t0 = time.time()
    filled_indices = np.argwhere(voxel_matrix > threshold)
    t1 = time.time()
    #print(f"Conversiontime voxel_matrix -> voxel_coords: {t1 - t0:.6f}s")

    # Scale indices by cube size to get voxel positions in space
    scaled_coords = filled_indices * cube_size

    # Adjust to get the center of each voxel
    centered_coords = scaled_coords + cube_size / 2

    # Apply translation to place voxels in the correct world position
    voxel_coords = centered_coords + translation

    # Extract velocity vectors for the occupied voxels
    t0 = time.time()
    velocity_vecs = velocity_matrix[filled_indices[:, 0], 
                    filled_indices[:, 1], 
                    filled_indices[:, 2]]
    t1 = time.time()
    print(f"Velocity vector extraction time: {t1 - t0:.6f}s")

    return voxel_coords, velocity_vecs
