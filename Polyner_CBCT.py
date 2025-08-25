# ----------------------------------------------#
# Pro    : cbct
# File   : Polyner_CBCT.py
# Date   : 2024/08/25
# Author : Modified for CBCT
# Email  : 
# ----------------------------------------------#
import model
import torch
import numpy as np
import dataset
import SimpleITK as sitk
import tinycudann as tcnn
from tqdm import tqdm
from torch.utils import data
from scipy import io as scio
from torch.optim import lr_scheduler
from skimage.morphology import erosion, square

def train_cbct(img_id, config):
    """
    CBCT-specific training function
    """
    # Data paths and parameters
    # -----------------------
    in_path = config["file"]["in_dir"]
    out_path = config["file"]["out_dir"]
    model_path = config["file"]["model_dir"]
    proj_path = '{}/ma_sinogram_{}.nii'.format(in_path, img_id)
    mask_path = '{}/mask_{}.nii'.format(in_path, img_id)
    
    # CBCT geometry parameters
    h, w, d = config["file"]["h"], config["file"]["w"], config["file"]["d"]
    SOD = config["file"]["SOD"]
    SDD = config["file"]["SDD"]
    detector_height = config["file"]["detector_height"]
    detector_width = config["file"]["detector_width"]
    num_projections = config["file"]["num_projections"]
    voxel_size = config["file"]["voxel_size"]

    # Training hyper-parameters
    # -----------------------
    lr = config["train"]["lr"]
    gpu = config["train"]["gpu"]
    epoch = config["train"]["epoch"]
    save_epoch = config["train"]["save_epoch"]
    lr_decay_epoch = config["train"]["lr_decay_epoch"]
    lr_decay_coefficient = config["train"]["lr_decay_coefficient"]
    batch_size = config["train"]["batch_size"]
    num_sample_ray = config["train"]["num_sample_ray"]
    lamb = config["train"]["lambda"]

    device = torch.device('cuda:{}'.format(str(gpu) if torch.cuda.is_available() else 'cpu'))

    # Load and process 3D mask
    # -----------------------
    try:
        mask_data = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        
        # Ensure mask is 3D
        if len(mask_data.shape) == 2:
            # 2D mask - extend to 3D
            mask_3d = np.tile(mask_data[np.newaxis, :, :], (d, 1, 1))
        else:
            mask_3d = mask_data
            
        # Pad mask to match volume size if needed
        if mask_3d.shape != (d, h, w):
            # Resize or pad mask to match volume dimensions
            pad_d = max(0, d - mask_3d.shape[0])
            pad_h = max(0, h - mask_3d.shape[1])
            pad_w = max(0, w - mask_3d.shape[2])
            
            mask_3d = np.pad(mask_3d, 
                           ((pad_d//2, pad_d//2 + pad_d%2),
                            (pad_h//2, pad_h//2 + pad_h%2), 
                            (pad_w//2, pad_w//2 + pad_w%2)))
            
            # Crop if larger
            mask_3d = mask_3d[:d, :h, :w]
        
        mask_3d = torch.tensor(mask_3d).float().unsqueeze(0).to(device)  # Add channel dim
        mask_3d = torch.where(mask_3d == 1, 0., 1.)  # Invert mask
        
    except Exception as e:
        print(f"Warning: Could not load mask {mask_path}. Using default mask. Error: {e}")
        mask_3d = torch.ones(1, d, h, w, device=device)

    # Energy spectrum
    # -----------------------
    spectrum_path = './{}/GE14Spectrum120KVP.mat'.format(in_path)
    try:
        spectrum = scio.loadmat(spectrum_path)['GE14Spectrum120KVP']
        e_1, e_n = 20, 120
        spectrum = spectrum[e_1-1:e_n, 1]
        spectrum = spectrum / np.sum(spectrum)
        e_level = len(spectrum)
        spectrum = torch.tensor(spectrum, dtype=torch.float).view(1, 1, 1, -1).to(device)
    except Exception as e:
        print(f"Warning: Could not load spectrum. Using default. Error: {e}")
        e_level = 101  # Default energy levels (20-120 keV)
        spectrum = torch.ones(1, 1, 1, e_level, device=device) / e_level

    # Model setup
    # -----------------------
    dc_loss = torch.nn.L1Loss().to(device)
    
    # Use CBCT-specific loss function
    cbct_loss = model.CBCT_Attenuation_Smoothness_Loss(
        mask_3d=mask_3d, lamb=lamb, spatial_weight=0.5, energy_weight=1.0
    ).to(device)

    # Network with 3D input
    network = tcnn.NetworkWithInputEncoding(
        n_input_dims=3,  # 3D coordinates (x, y, z)
        n_output_dims=e_level,
        encoding_config=config["encoding"], 
        network_config=config["network"]
    ).to(device)
    
    optimizer = torch.optim.Adam(params=network.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_epoch, gamma=lr_decay_coefficient)

    # CBCT geometry configuration
    geometry_config = {
        'SOD': SOD,
        'SDD': SDD,
        'detector_height': detector_height,
        'detector_width': detector_width,
        'num_projections': num_projections
    }

    # Data loaders
    # -----------------------
    train_loader = data.DataLoader(
        dataset=dataset.CBCTTrainData(
            proj_path=proj_path,
            geometry_config=geometry_config,
            num_sample_ray=num_sample_ray,
            voxel_size=voxel_size,
            volume_depth=d
        ),
        batch_size=batch_size, 
        shuffle=True
    )
    
    test_loader = data.DataLoader(
        dataset=dataset.CBCTTestData(h=h, w=w, d=d),
        batch_size=1, 
        shuffle=False
    )

    # Training loop
    # -----------------------
    loop_tqdm = tqdm(range(epoch), leave=False)
    for e in loop_tqdm:
        network.train()
        loss_log = 0
        
        for i, (ray, proj) in enumerate(train_loader):
            ray = ray.to(device).float()  # (batch_size, num_sample_ray, volume_depth, 3)
            proj = proj.to(device).float()  # (batch_size, num_sample_ray)
            
            batch_size, num_sample_ray, volume_depth, _ = ray.shape
            
            # Reshape for network input
            ray_flat = ray.view(-1, 3)  # (batch_size*num_sample_ray*volume_depth, 3)
            
            # Forward pass through network
            intensity_pre = network(ray_flat).view(batch_size, num_sample_ray, volume_depth, e_level)
            
            # CBCT forward model (line integral along ray)
            proj_pre = torch.exp(-voxel_size * torch.sum(intensity_pre, dim=2))  # (batch_size, num_sample_ray, e_level)
            
            # Apply energy spectrum
            proj_pre = -torch.log(torch.sum(proj_pre * spectrum.squeeze(), dim=-1))  # (batch_size, num_sample_ray)
            
            # Calculate losses
            data_loss = dc_loss(proj_pre, proj.to(proj_pre.dtype))
            regularization_loss = cbct_loss(intensity=intensity_pre, ray=ray)
            
            total_loss = data_loss + regularization_loss

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            loss_log += total_loss.item()
        
        scheduler.step()
        loop_tqdm.set_description(f"CBCT Image #{img_id}")
        loop_tqdm.set_postfix(
            lr=scheduler.get_last_lr()[0], 
            loss=loss_log / len(train_loader)
        )

        # Model save & 3D reconstruction
        if (e + 1) % save_epoch == 0:
            with torch.no_grad():
                torch.save(network.state_dict(), f'{model_path}/cbct_model_{img_id}.pkl')
                
                # Reconstruct 3D volume
                for i, (xyz) in enumerate(test_loader):
                    xyz = xyz.to(device).float().view(-1, 3)  # (h*w*d, 3)
                    
                    # Process in batches to avoid memory issues
                    chunk_size = 10000
                    volume_pred = []
                    
                    for start_idx in range(0, xyz.shape[0], chunk_size):
                        end_idx = min(start_idx + chunk_size, xyz.shape[0])
                        xyz_chunk = xyz[start_idx:end_idx]
                        
                        # Get attenuation at mean energy
                        mean_energy_idx = e_level // 2
                        chunk_pred = network(xyz_chunk)[:, mean_energy_idx]
                        volume_pred.append(chunk_pred.cpu().detach().numpy())
                    
                    volume_pred = np.concatenate(volume_pred)
                    volume_3d = volume_pred.reshape(d, h, w)
                    
                    # Save 3D volume
                    volume_sitk = sitk.GetImageFromArray(volume_3d)
                    sitk.WriteImage(volume_sitk, f'{out_path}/cbct_polyner_{img_id}.nii')
                    
                    print(f"Saved 3D CBCT reconstruction: {out_path}/cbct_polyner_{img_id}.nii")


def train_legacy(img_id, config):
    """
    Legacy fan-beam training for backward compatibility
    """
    # Import and use original training logic
    import Polyner
    return Polyner.train(img_id, config)


def train(img_id, config, projection_mode='cbct'):
    """
    Main training function that dispatches to appropriate implementation
    """
    mode = config["file"].get("projection_mode", projection_mode)
    
    if mode == 'cbct':
        return train_cbct(img_id, config)
    else:
        return train_legacy(img_id, config)