# ----------------------------------------------#
# Pro    : cbct
# File   : dataset.py
# Date   : 2023/2/22
# Author : Qing Wu
# Email  : wuqing@shanghaitech.edu.cn
# ----------------------------------------------#
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attenuation_Smootion_Over_Energies_Loss(nn.Module):
    def __init__(self, mask, lamb, projection_mode='fan_beam'):
        super(Attenuation_Smootion_Over_Energies_Loss, self).__init__()
        self.mask = mask
        self.lamb = lamb
        self.projection_mode = projection_mode
        
    def forward(self, ray, intensity):
        if self.projection_mode == 'cbct':
            # Handle 3D CBCT case
            batch_size, num_sample_ray, k, e_level = intensity.shape
            
            # For CBCT, ray coordinates are 3D (x, y, z)
            # Extract x, y coordinates for mask sampling (assuming mask is 2D)
            ray_2d = ray[:, :, :, :2]  # (batch_size, num_sample_ray, k, 2)
            
            # Sample mask using 2D coordinates
            mask = F.grid_sample(
                self.mask, ray_2d.view(-1, 1, k, 2), mode='nearest', align_corners=False
            ).view(batch_size, num_sample_ray, k)   # (batch_size, num_sample_ray, k)
            
            # Energy smoothness constraint
            diff = torch.sum(torch.abs(intensity[:, :, :, 1:] - intensity[:, :, :, :e_level-1]), dim=-1) * mask
            return self.lamb * torch.sum(diff) / (batch_size * num_sample_ray * k)
        
        else:
            # Legacy fan-beam case
            batch_size, num_sample_ray, k, e_level = intensity.shape
            mask = F.grid_sample(
                self.mask, ray.unsqueeze(0).unsqueeze(0), mode='nearest', align_corners=False
            )[0, 0, 0, :].view(batch_size, num_sample_ray, k)   # (batch_size, num_sample_ray, 2*SOD)
            diff = torch.sum(torch.abs(intensity[:, :, :, 1:] - intensity[:, :, :, :e_level-1]), dim=-1) * mask
            return self.lamb * torch.sum(diff) / (batch_size * num_sample_ray * k)


class CBCT_Attenuation_Smoothness_Loss(nn.Module):
    """
    Dedicated CBCT loss function with 3D volume constraints
    """
    def __init__(self, mask_3d, lamb, spatial_weight=1.0, energy_weight=1.0):
        super(CBCT_Attenuation_Smoothness_Loss, self).__init__()
        self.mask_3d = mask_3d
        self.lamb = lamb
        self.spatial_weight = spatial_weight
        self.energy_weight = energy_weight
        
    def forward(self, ray, intensity):
        """
        Args:
            ray: (batch_size, num_sample_ray, volume_depth, 3)
            intensity: (batch_size, num_sample_ray, volume_depth, e_level)
        """
        batch_size, num_sample_ray, volume_depth, e_level = intensity.shape
        
        # Sample 3D mask
        if self.mask_3d is not None:
            # ray coordinates are in [-1, 1] range
            mask = F.grid_sample(
                self.mask_3d.unsqueeze(0).unsqueeze(0),  # Add batch and channel dims
                ray.view(1, -1, 1, 1, 3),  # Reshape for 3D sampling
                mode='nearest', align_corners=False
            ).view(batch_size, num_sample_ray, volume_depth)
        else:
            mask = torch.ones(batch_size, num_sample_ray, volume_depth, 
                            device=intensity.device, dtype=intensity.dtype)
        
        total_loss = 0.0
        
        # Energy smoothness constraint
        if self.energy_weight > 0:
            energy_diff = torch.sum(torch.abs(intensity[:, :, :, 1:] - intensity[:, :, :, :e_level-1]), dim=-1)
            energy_loss = self.energy_weight * torch.sum(energy_diff * mask) / (batch_size * num_sample_ray * volume_depth)
            total_loss += energy_loss
        
        # Spatial smoothness constraint (along ray direction)
        if self.spatial_weight > 0:
            spatial_diff = torch.sum(torch.abs(intensity[:, :, 1:, :] - intensity[:, :, :-1, :]), dim=-1)
            mask_spatial = mask[:, :, 1:]  # Adjust mask size for spatial difference
            spatial_loss = self.spatial_weight * torch.sum(spatial_diff * mask_spatial) / (batch_size * num_sample_ray * (volume_depth-1))
            total_loss += spatial_loss
        
        return self.lamb * total_loss
