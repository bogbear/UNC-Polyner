# ----------------------------------------------#
# Pro    : cbct
# File   : dataset.py
# Date   : 2023/2/22
# Author : Qing Wu
# Email  : wuqing@shanghaitech.edu.cn
# ----------------------------------------------#
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio


def psnr(image, ground_truth):
    data_range = np.max(ground_truth) - np.min(ground_truth)
    return peak_signal_noise_ratio(ground_truth, image, data_range=data_range)


def ssim(image, ground_truth):
    data_range = np.max(ground_truth) - np.min(ground_truth)
    return structural_similarity(image, ground_truth, data_range=data_range)


def cone_beam_ray(det_u, det_v, SOD, SDD, volume_size):
    """
    Generate cone-beam rays for CBCT geometry
    
    Args:
        det_u: detector u coordinates (horizontal)
        det_v: detector v coordinates (vertical)  
        SOD: source-to-object distance
        SDD: source-to-detector distance
        volume_size: size of reconstruction volume
        
    Returns:
        rays: (num_det_u, num_det_v, volume_size, 3) array of 3D coordinates along each ray
    """
    num_det_u = len(det_u)
    num_det_v = len(det_v)
    
    # Source position at origin (0, -SOD, 0)
    source_pos = np.array([0, -SOD, 0])
    
    # Volume sampling points along z-axis (depth)
    z_samples = np.linspace(-1, 1, volume_size)
    
    rays = np.zeros((num_det_u, num_det_v, volume_size, 3))
    
    for i, u in enumerate(det_u):
        for j, v in enumerate(det_v):
            # Detector pixel position
            det_pos = np.array([u, SDD - SOD, v])
            
            # Ray direction from source to detector pixel
            ray_dir = det_pos - source_pos
            ray_dir = ray_dir / np.linalg.norm(ray_dir)
            
            # Sample points along the ray through the volume
            for k, z in enumerate(z_samples):
                # Parametric ray equation: point = source + t * direction
                # We sample uniformly in the volume coordinate system
                t = (z + SOD) / ray_dir[1] if ray_dir[1] != 0 else 0
                point = source_pos + t * ray_dir
                
                # Normalize to [-1, 1] coordinate system
                rays[i, j, k] = point / SOD
    
    return rays


def fan_beam_ray(proj_pos, SOD):
    """Legacy fan-beam ray generation for backward compatibility"""
    origin_x = 0
    origin_y = -1
    y = np.linspace(-1, 1, int(2*SOD)).reshape(-1, 1)  # (2*SOD, ) -> (2*SOD, 1)
    x = np.zeros_like(y)  # (2*SOD, 1)
    xy_temp = np.concatenate((x, y), axis=-1)  # (2*SOD, 2)
    xy_temp = np.concatenate((xy_temp, np.ones_like(x)), axis=-1)  # (2*SOD, 3)
    num_det = len(proj_pos)
    xy = np.zeros(shape=(num_det, int(2*SOD), 2)) # (L, 2*SOD, 2)
    for i in range(num_det):
        fan_angle_rad = np.deg2rad(proj_pos[num_det-i-1])
        M = np.array(
            [
                [np.cos(fan_angle_rad), -np.sin(fan_angle_rad),
                 -1*origin_x*np.cos(fan_angle_rad)+origin_y*np.sin(fan_angle_rad)+origin_x],
                [np.sin(fan_angle_rad), np.cos(fan_angle_rad),
                 -1*origin_x*np.sin(fan_angle_rad)-origin_y*np.cos(fan_angle_rad)+origin_y],
                [0, 0, 1]
            ]
        )
        temp = xy_temp @ M.T # (2*SOD, 3) @ (3, 3) -> (2*SOD, 3)
        xy[i, :, :] = temp[:, :2] # (2*SOD, 2)
    return xy


def grid_coordinate_3d(h, w, d):
    """Generate 3D grid coordinates for volume reconstruction"""
    x = np.linspace(-1, 1, h)
    y = np.linspace(-1, 1, w)
    z = np.linspace(-1, 1, d)
    x, y, z = np.meshgrid(x, y, z, indexing='ij')  # (h, w, d), (h, w, d), (h, w, d)
    xyz = np.stack([x, y, z], -1).reshape(-1, 3)  # (h*w*d, 3)
    return xyz


def grid_coordinate(h, w):
    """Legacy 2D grid coordinate generation for backward compatibility"""
    x = np.linspace(-1, 1, h)
    y = np.linspace(-1, 1, w)
    x, y = np.meshgrid(x, y, indexing='ij')  # (h, w), (h, w)
    xy = np.stack([x, y], -1).reshape(-1, 2)  # (h*w, 2)
    return xy


def rotate_ray_3d(xyz, angle, axis='z'):
    """
    Rotate 3D rays around specified axis
    
    Args:
        xyz: input coordinates (..., 3)
        angle: rotation angle in degrees
        axis: rotation axis ('x', 'y', or 'z')
    """
    xyz_shape = xyz.shape
    angle_rad = np.deg2rad(angle)
    
    if axis == 'z':
        # Rotation around z-axis (most common for CBCT)
        trans_mat = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad),  np.cos(angle_rad), 0],
            [0, 0, 1]
        ])
    elif axis == 'y':
        # Rotation around y-axis
        trans_mat = np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad)],
            [0, 1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ])
    elif axis == 'x':
        # Rotation around x-axis
        trans_mat = np.array([
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad), np.cos(angle_rad)]
        ])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")
    
    xyz = xyz.reshape(-1, 3)
    xyz = (np.dot(xyz, trans_mat.T)).reshape(xyz_shape)
    return xyz


def create_cbct_geometry(detector_height, detector_width, SOD, SDD, num_angles):
    """
    Create CBCT geometry parameters
    
    Args:
        detector_height: number of detector pixels in v direction (vertical)
        detector_width: number of detector pixels in u direction (horizontal)
        SOD: source-to-object distance
        SDD: source-to-detector distance
        num_angles: number of projection angles
        
    Returns:
        dict containing geometry parameters
    """
    # Detector pixel coordinates (centered at origin)
    det_u = np.linspace(-1, 1, detector_width)
    det_v = np.linspace(-1, 1, detector_height)
    
    # Projection angles (360 degrees)
    angles = np.linspace(0., 360., num=num_angles, endpoint=False)
    
    return {
        'det_u': det_u,
        'det_v': det_v,
        'angles': angles,
        'SOD': SOD,
        'SDD': SDD,
        'detector_height': detector_height,
        'detector_width': detector_width
    }


def cbct_forward_project(volume, geometry, voxel_size, spectrum):
    """
    Forward projection for CBCT geometry
    
    Args:
        volume: 3D volume data
        geometry: geometry parameters
        voxel_size: size of voxels
        spectrum: energy spectrum
        
    Returns:
        projected data
    """
    # This is a placeholder for the forward projection
    # In practice, this would integrate along ray paths
    pass


def cbct_ray_sampling(det_u_idx, det_v_idx, angle, geometry, volume_depth):
    """
    Generate ray samples for a specific detector pixel and projection angle
    
    Args:
        det_u_idx: detector u index
        det_v_idx: detector v index  
        angle: projection angle in degrees
        geometry: geometry parameters
        volume_depth: number of samples along ray depth
        
    Returns:
        ray_coords: (volume_depth, 3) coordinates along the ray
    """
    SOD = geometry['SOD']
    SDD = geometry['SDD']
    det_u = geometry['det_u'][det_u_idx]
    det_v = geometry['det_v'][det_v_idx]
    
    # Source position (rotated)
    angle_rad = np.deg2rad(angle)
    source_pos = np.array([
        SOD * np.sin(angle_rad),
        -SOD * np.cos(angle_rad), 
        0
    ])
    
    # Detector pixel position (rotated)
    det_pos_local = np.array([det_u, SDD - SOD, det_v])
    det_pos = np.array([
        det_pos_local[0] * np.cos(angle_rad) + det_pos_local[1] * np.sin(angle_rad),
        -det_pos_local[0] * np.sin(angle_rad) + det_pos_local[1] * np.cos(angle_rad),
        det_pos_local[2]
    ])
    
    # Ray direction
    ray_dir = det_pos - source_pos
    ray_dir = ray_dir / np.linalg.norm(ray_dir)
    
    # Sample points along ray through volume
    t_values = np.linspace(0, 2 * SOD, volume_depth)
    ray_coords = source_pos[np.newaxis, :] + t_values[:, np.newaxis] * ray_dir[np.newaxis, :]
    
    # Normalize to [-1, 1] coordinate system
    ray_coords = ray_coords / SOD
    
    return ray_coords


def rotate_ray(xy, angle):
    """Legacy 2D ray rotation for backward compatibility"""
    xy_shape = xy.shape
    angle_rad = np.deg2rad(angle)
    trans_mat = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad),  np.cos(angle_rad)],
        ]
    )
    xy = xy.reshape(-1, 2)
    xy = (np.dot(xy, trans_mat.T)).reshape(xy_shape)
    return xy