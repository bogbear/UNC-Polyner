# ----------------------------------------------#
# Pro    : cbct
# File   : dataset.py
# Date   : 2023/2/22
# Author : Qing Wu
# Email  : wuqing@shanghaitech.edu.cn
# ----------------------------------------------#
import utils
import numpy as np
import SimpleITK as sitk
from torch.utils import data

class TrainData(data.Dataset):
    def __init__(self, proj_path, proj_pos_path, num_sample_ray, num_angle, SOD, voxel_size, 
                 projection_mode='fan_beam', **kwargs):
        self.num_angle = num_angle
        self.num_sample_ray = num_sample_ray
        self.SOD = SOD
        self.voxel_size = voxel_size
        self.projection_mode = projection_mode
        self.angles = np.linspace(0., 360., num=self.num_angle, endpoint=False)  # (num_angle, )
        
        if projection_mode == 'cbct':
            # CBCT mode - handle 3D projection data
            self.SDD = kwargs.get('SDD', 652)
            self.detector_height = kwargs.get('detector_height', 400)
            self.detector_width = kwargs.get('detector_width', 400)
            self.volume_depth = kwargs.get('volume_depth', 2*SOD)
            
            # Create CBCT geometry
            self.geometry = utils.create_cbct_geometry(
                self.detector_height, self.detector_width, self.SOD, self.SDD, self.num_angle
            )
            
            # Load 3D projection data (num_angles, detector_height, detector_width)
            proj_data = sitk.GetArrayFromImage(sitk.ReadImage(proj_path))
            if len(proj_data.shape) == 2:
                # Convert 2D sinogram to 3D format (assume single slice)
                self.proj = proj_data[:, np.newaxis, :]  # (num_angle, 1, num_det)
                self.detector_height = 1
            else:
                self.proj = proj_data  # (num_angle, detector_height, detector_width)
            
            self.num_det_u = self.detector_width
            self.num_det_v = self.detector_height
            self.index_max_u = self.num_det_u - self.num_sample_ray
            self.index_max_v = max(1, self.num_det_v - 1)
            
        else:
            # Legacy fan-beam mode
            self.proj_pos = sitk.GetArrayFromImage(sitk.ReadImage(proj_pos_path)).reshape(-1) # (num_det, )
            self.num_det = len(self.proj_pos)
            # projection, i.e., sinogram & metal_trace
            self.proj = sitk.GetArrayFromImage(sitk.ReadImage(proj_path))  # (num_angle, num_det)
            # ray
            self.rays = utils.fan_beam_ray(self.proj_pos, self.SOD) # (num_det, 2*SOD, 2)
            self.index_max = self.num_det - self.num_sample_ray

    def __getitem__(self, item):
        ang = self.angles[item]
        
        if self.projection_mode == 'cbct':
            # CBCT mode - sample detector pixels and generate 3D rays
            proj = self.proj[item]  # (detector_height, detector_width)
            
            # Sample detector pixels
            if self.detector_height > 1:
                index_v = np.random.randint(0, self.index_max_v, size=1)[0]
            else:
                index_v = 0
            index_u = np.random.randint(0, self.index_max_u, size=1)[0]
            
            ray_samples = []
            proj_samples = []
            
            for i in range(self.num_sample_ray):
                det_u_idx = index_u + i
                det_v_idx = index_v
                
                # Generate 3D ray coordinates
                ray_coords = utils.cbct_ray_sampling(
                    det_u_idx, det_v_idx, ang, self.geometry, self.volume_depth
                )
                
                ray_samples.append(ray_coords)
                
                # Get corresponding projection value
                if len(proj.shape) == 1:
                    proj_val = proj[det_u_idx]
                else:
                    proj_val = proj[det_v_idx, det_u_idx]
                proj_samples.append(proj_val)
            
            ray_sample = np.stack(ray_samples)  # (num_sample_ray, volume_depth, 3)
            proj_sample = np.array(proj_samples)  # (num_sample_ray,)
            
            return ray_sample, proj_sample
        
        else:
            # Legacy fan-beam mode
            proj = self.proj[item].reshape(-1, )  # (num_det, )
            # sample ray, projection, and metal trace
            index = np.random.randint(0, self.index_max, size=1)[0]
            ray_sample = self.rays[index:index+self.num_sample_ray]     # (num_sample_ray, 2*SOD, 2)
            proj_sample = proj[index:index+self.num_sample_ray]     # (num_sample_ray, )
            # rotate ray
            ray_sample = utils.rotate_ray(xy=ray_sample, angle=ang)
            return ray_sample, proj_sample

    def __len__(self):
        return self.num_angle


class TestData(data.Dataset):
    def __init__(self, h, w, d=None, projection_mode='fan_beam'):
        self.h, self.w = h, w
        self.d = d
        self.projection_mode = projection_mode
        
        if projection_mode == 'cbct' and d is not None:
            # 3D CBCT mode
            self.xyz = utils.grid_coordinate_3d(h=self.h, w=self.w, d=self.d).reshape(1, int(h*w*d), 3)
        else:
            # Legacy 2D fan-beam mode
            self.xy = utils.grid_coordinate(h=self.h, w=self.w).reshape(1, int(h*w), 2)

    def __getitem__(self, item):
        if self.projection_mode == 'cbct' and self.d is not None:
            return self.xyz[item]    # (h*w*d, 3)
        else:
            return self.xy[item]    # (h*w, 2)

    def __len__(self):
        return 1


class CBCTTrainData(data.Dataset):
    """
    Dedicated CBCT training dataset for cleaner implementation
    """
    def __init__(self, proj_path, geometry_config, num_sample_ray, voxel_size, volume_depth=None):
        self.num_sample_ray = num_sample_ray
        self.voxel_size = voxel_size
        
        # Load geometry parameters
        self.SOD = geometry_config['SOD']
        self.SDD = geometry_config['SDD']
        self.detector_height = geometry_config['detector_height']
        self.detector_width = geometry_config['detector_width']
        self.num_projections = geometry_config['num_projections']
        
        # Set volume depth
        self.volume_depth = volume_depth if volume_depth else 2 * self.SOD
        
        # Create geometry
        self.geometry = utils.create_cbct_geometry(
            self.detector_height, self.detector_width, 
            self.SOD, self.SDD, self.num_projections
        )
        
        # Load projection data
        proj_data = sitk.GetArrayFromImage(sitk.ReadImage(proj_path))
        
        if len(proj_data.shape) == 2:
            # 2D sinogram - expand to 3D
            self.proj = proj_data[:, np.newaxis, :]  # (num_angles, 1, detector_width)
            self.detector_height = 1
        elif len(proj_data.shape) == 3:
            # 3D projection stack
            self.proj = proj_data  # (num_angles, detector_height, detector_width)
        else:
            raise ValueError(f"Unsupported projection data shape: {proj_data.shape}")
        
        # Update geometry with actual dimensions
        self.geometry['detector_height'] = self.detector_height
        
        # Sampling constraints
        self.index_max_u = max(1, self.detector_width - self.num_sample_ray)
        self.index_max_v = max(1, self.detector_height)
        
    def __getitem__(self, item):
        angle = self.geometry['angles'][item]
        proj = self.proj[item]  # (detector_height, detector_width) or (1, detector_width)
        
        # Sample detector pixels
        det_v_idx = np.random.randint(0, self.index_max_v) if self.detector_height > 1 else 0
        det_u_start = np.random.randint(0, self.index_max_u)
        
        ray_samples = []
        proj_samples = []
        
        for i in range(self.num_sample_ray):
            det_u_idx = det_u_start + i
            
            # Generate ray coordinates for this detector pixel
            ray_coords = utils.cbct_ray_sampling(
                det_u_idx, det_v_idx, angle, self.geometry, self.volume_depth
            )
            ray_samples.append(ray_coords)
            
            # Get projection value
            if len(proj.shape) == 1:
                proj_val = proj[det_u_idx]
            else:
                proj_val = proj[det_v_idx, det_u_idx]
            proj_samples.append(proj_val)
        
        ray_sample = np.stack(ray_samples)  # (num_sample_ray, volume_depth, 3)
        proj_sample = np.array(proj_samples)  # (num_sample_ray,)
        
        return ray_sample, proj_sample
        
    def __len__(self):
        return self.num_projections


class CBCTTestData(data.Dataset):
    """
    Dedicated CBCT test dataset for 3D volume reconstruction
    """
    def __init__(self, h, w, d):
        self.h, self.w, self.d = h, w, d
        self.xyz = utils.grid_coordinate_3d(h=h, w=w, d=d).reshape(1, h*w*d, 3)
        
    def __getitem__(self, item):
        return self.xyz[item]  # (h*w*d, 3)
        
    def __len__(self):
        return 1