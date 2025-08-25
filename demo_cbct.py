#!/usr/bin/env python3
# ----------------------------------------------#
# Pro    : cbct
# File   : demo_cbct.py
# Date   : 2024/08/25
# Author : CBCT Demo Script
# Email  : 
# ----------------------------------------------#
"""
Demo script to showcase CBCT Polyner functionality
This script demonstrates the key features and usage patterns
"""

import numpy as np
import torch
import SimpleITK as sitk
import os
import json
from scipy.io import savemat
import argparse

# Import our modules
import utils
import dataset
import model

def create_synthetic_cbct_data(output_dir="./demo_data"):
    """
    Create synthetic CBCT data for demonstration
    """
    print("Creating synthetic CBCT data...")
    os.makedirs(output_dir, exist_ok=True)
    
    # CBCT geometry parameters
    SOD = 362
    SDD = 652
    detector_height = 128
    detector_width = 128  
    num_projections = 180
    volume_size = 64  # Smaller for demo
    
    # Create synthetic 3D volume with metal insert
    print("  - Generating 3D phantom volume...")
    phantom = np.zeros((volume_size, volume_size, volume_size))
    
    # Add soft tissue background
    phantom[:, :, :] = 0.02  # Soft tissue attenuation
    
    # Add metal insert (high attenuation)
    center = volume_size // 2
    metal_size = 8
    phantom[center-metal_size:center+metal_size, 
           center-metal_size:center+metal_size,
           center-metal_size:center+metal_size] = 0.5  # Metal attenuation
    
    # Save ground truth volume
    gt_volume = sitk.GetImageFromArray(phantom)
    sitk.WriteImage(gt_volume, os.path.join(output_dir, "gt_0.nii"))
    
    # Create metal mask  
    print("  - Creating metal mask...")
    mask = np.zeros_like(phantom)
    mask[center-metal_size:center+metal_size, 
         center-metal_size:center+metal_size,
         center-metal_size:center+metal_size] = 1
    mask_volume = sitk.GetImageFromArray(mask)
    sitk.WriteImage(mask_volume, os.path.join(output_dir, "mask_0.nii"))
    
    # Create CBCT geometry
    print("  - Setting up CBCT geometry...")
    geometry = utils.create_cbct_geometry(
        detector_height, detector_width, SOD, SDD, num_projections
    )
    
    # Simulate CBCT projections (simplified forward projection)
    print("  - Simulating CBCT projections...")
    projections = np.zeros((num_projections, detector_width))  # 2D sinogram for simplicity
    
    for angle_idx, angle in enumerate(geometry['angles']):
        # Simple ray-driven forward projection
        for det_idx in range(detector_width):
            # Get ray path
            ray_coords = utils.cbct_ray_sampling(
                det_idx, 0, angle, geometry, volume_size
            )
            
            # Sample phantom along ray (trilinear interpolation)
            ray_coords_scaled = (ray_coords + 1) * (volume_size - 1) / 2  # Scale to volume indices
            
            # Clamp coordinates
            ray_coords_scaled = np.clip(ray_coords_scaled, 0, volume_size - 1)
            
            # Simple nearest neighbor sampling
            indices = np.round(ray_coords_scaled).astype(int)
            
            # Integrate along ray
            line_integral = 0
            for i, (x, y, z) in enumerate(indices):
                if 0 <= x < volume_size and 0 <= y < volume_size and 0 <= z < volume_size:
                    line_integral += phantom[z, y, x]  # Note: SimpleITK uses z,y,x order
            
            # Add noise and metal artifacts
            line_integral *= 0.1  # Voxel size scaling
            projection_value = np.exp(-line_integral)
            
            # Add metal artifacts (simplified)
            if line_integral > 0.3:  # Ray passes through metal
                projection_value *= 0.1  # Strong attenuation
                
            projections[angle_idx, det_idx] = -np.log(max(projection_value, 1e-6))
    
    # Add noise
    projections += np.random.normal(0, 0.01, projections.shape)
    
    # Save projection data
    proj_volume = sitk.GetImageFromArray(projections)
    sitk.WriteImage(proj_volume, os.path.join(output_dir, "ma_sinogram_0.nii"))
    
    # Create energy spectrum
    print("  - Creating energy spectrum...")
    energies = np.arange(20, 121)  # 20-120 keV
    spectrum = np.exp(-(energies - 60)**2 / (2 * 15**2))  # Gaussian spectrum centered at 60 keV
    spectrum = spectrum / np.sum(spectrum)  # Normalize
    
    spectrum_data = np.column_stack([energies, spectrum])
    spectrum_path = os.path.join(output_dir, "GE14Spectrum120KVP.mat")
    savemat(spectrum_path, {"GE14Spectrum120KVP": spectrum_data})
    
    print(f"Synthetic CBCT data created in: {output_dir}")
    return output_dir

def demo_geometry_functions():
    """
    Demonstrate CBCT geometry functions
    """
    print("\n" + "="*50)
    print("CBCT GEOMETRY DEMONSTRATION")  
    print("="*50)
    
    # Create CBCT geometry
    SOD = 362
    SDD = 652
    detector_height = 100
    detector_width = 100
    num_angles = 180
    
    geometry = utils.create_cbct_geometry(
        detector_height, detector_width, SOD, SDD, num_angles
    )
    
    print(f"Geometry parameters:")
    print(f"  SOD: {geometry['SOD']} mm")
    print(f"  SDD: {geometry['SDD']} mm") 
    print(f"  Detector: {detector_height} × {detector_width} pixels")
    print(f"  Projections: {num_angles} angles")
    
    # Demonstrate ray sampling
    print(f"\nRay sampling example:")
    ray_coords = utils.cbct_ray_sampling(50, 50, 0, geometry, 100)
    print(f"  Ray coordinates shape: {ray_coords.shape}")
    print(f"  Ray start: {ray_coords[0]}")
    print(f"  Ray end: {ray_coords[-1]}")
    
    # Demonstrate 3D rotation
    print(f"\n3D rotation example:")
    test_point = np.array([[[1, 0, 0]]])
    rotated = utils.rotate_ray_3d(test_point, 90, axis='z')
    print(f"  Original: {test_point.flatten()}")
    print(f"  Rotated 90° around z: {rotated.flatten()}")

def demo_dataset_creation(data_dir):
    """
    Demonstrate CBCT dataset creation and usage
    """
    print("\n" + "="*50)
    print("CBCT DATASET DEMONSTRATION")
    print("="*50)
    
    # CBCT geometry configuration
    geometry_config = {
        'SOD': 362,
        'SDD': 652,
        'detector_height': 64,  # Smaller for demo
        'detector_width': 128,
        'num_projections': 180
    }
    
    # Create CBCT training dataset
    proj_path = os.path.join(data_dir, 'ma_sinogram_0.nii')
    
    if os.path.exists(proj_path):
        print("Creating CBCT training dataset...")
        train_dataset = dataset.CBCTTrainData(
            proj_path=proj_path,
            geometry_config=geometry_config,
            num_sample_ray=4,
            voxel_size=0.1,
            volume_depth=64
        )
        
        print(f"  Dataset length: {len(train_dataset)}")
        
        # Sample from dataset
        ray_sample, proj_sample = train_dataset[0]
        print(f"  Ray sample shape: {ray_sample.shape}")
        print(f"  Projection sample shape: {proj_sample.shape}")
        print(f"  Ray coordinate range: [{ray_sample.min():.2f}, {ray_sample.max():.2f}]")
        
        # Create CBCT test dataset
        print("\nCreating CBCT test dataset...")
        test_dataset = dataset.CBCTTestData(h=64, w=64, d=64)
        
        xyz = test_dataset[0]
        print(f"  Test coordinates shape: {xyz.shape}")
        print(f"  Coordinate range: [{xyz.min():.2f}, {xyz.max():.2f}]")
    else:
        print(f"Projection data not found: {proj_path}")

def demo_loss_functions():
    """
    Demonstrate CBCT loss functions
    """
    print("\n" + "="*50)
    print("CBCT LOSS FUNCTIONS DEMONSTRATION")
    print("="*50)
    
    # Create synthetic data
    batch_size = 2
    num_sample_ray = 4
    volume_depth = 32
    e_level = 101
    
    device = torch.device('cpu')
    
    # Create synthetic 3D mask
    mask_3d = torch.ones(1, 32, 32, 32, device=device)
    mask_3d[:, 10:20, 10:20, 10:20] = 0  # Metal region
    
    # Create CBCT loss function
    cbct_loss = model.CBCT_Attenuation_Smoothness_Loss(
        mask_3d=mask_3d, 
        lamb=0.2, 
        spatial_weight=0.5, 
        energy_weight=1.0
    )
    
    # Create synthetic input data
    ray = torch.randn(batch_size, num_sample_ray, volume_depth, 3, device=device)
    intensity = torch.randn(batch_size, num_sample_ray, volume_depth, e_level, device=device)
    
    # Compute loss
    loss_value = cbct_loss(ray, intensity)
    
    print(f"CBCT Loss Function:")
    print(f"  Input ray shape: {ray.shape}")
    print(f"  Input intensity shape: {intensity.shape}")
    print(f"  Loss value: {loss_value.item():.4f}")
    
    # Compare with legacy loss
    mask_2d = torch.ones(1, 1, 64, 64, device=device)
    legacy_loss = model.Attenuation_Smootion_Over_Energies_Loss(
        mask=mask_2d, lamb=0.2, projection_mode='cbct'
    )
    
    legacy_loss_value = legacy_loss(ray, intensity)
    print(f"\nLegacy Loss (CBCT mode):")
    print(f"  Loss value: {legacy_loss_value.item():.4f}")

def demo_configuration():
    """
    Demonstrate CBCT configuration setup
    """
    print("\n" + "="*50)
    print("CBCT CONFIGURATION DEMONSTRATION")
    print("="*50)
    
    # Create sample CBCT configuration
    cbct_config = {
        "file": {
            "in_dir": "./demo_data",
            "model_dir": "./demo_models",
            "out_dir": "./demo_output",
            "voxel_size": 0.1,
            "SOD": 362,
            "SDD": 652,
            "h": 128,
            "w": 128,
            "d": 128,
            "detector_height": 256,
            "detector_width": 256,
            "num_projections": 360,
            "projection_mode": "cbct"
        },
        "train": {
            "gpu": 0,
            "lr": 1e-3,
            "epoch": 2000,
            "save_epoch": 1000,
            "num_sample_ray": 4,
            "lr_decay_epoch": 1000,
            "lr_decay_coefficient": 0.1,
            "batch_size": 8,
            "lambda": 0.2
        },
        "encoding": {
            "otype": "Grid",
            "type": "Hash",
            "n_levels": 16,
            "n_features_per_level": 8,
            "log2_hashmap_size": 19,
            "base_resolution": 2,
            "per_level_scale": 2,
            "interpolation": "Linear"
        },
        "network": {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "Squareplus",
            "n_neurons": 128,
            "n_hidden_layers": 3,
            "input_dims": 3
        }
    }
    
    print("Sample CBCT Configuration:")
    print(json.dumps(cbct_config, indent=2))
    
    # Save configuration
    config_path = "./demo_cbct_config.json"
    with open(config_path, 'w') as f:
        json.dump(cbct_config, f, indent=2)
    
    print(f"\nConfiguration saved to: {config_path}")

def main():
    parser = argparse.ArgumentParser(description='CBCT Polyner Demo')
    parser.add_argument('--create-data', action='store_true', 
                      help='Create synthetic CBCT data')
    parser.add_argument('--demo-all', action='store_true',
                      help='Run all demonstrations')
    parser.add_argument('--data-dir', type=str, default='./demo_data',
                      help='Directory for demo data')
    
    args = parser.parse_args()
    
    print("🏥 CBCT POLYNER DEMONSTRATION")
    print("="*60)
    print("This script demonstrates the key features of CBCT Polyner")
    print("="*60)
    
    # Create synthetic data if requested or needed
    if args.create_data or args.demo_all or not os.path.exists(args.data_dir):
        data_dir = create_synthetic_cbct_data(args.data_dir)
    else:
        data_dir = args.data_dir
    
    if args.demo_all:
        # Run all demonstrations
        demo_geometry_functions()
        demo_dataset_creation(data_dir)
        demo_loss_functions() 
        demo_configuration()
    else:
        # Interactive menu
        while True:
            print("\n" + "="*40)
            print("DEMO OPTIONS")
            print("="*40)
            print("1. Geometry Functions")
            print("2. Dataset Creation")
            print("3. Loss Functions")
            print("4. Configuration")
            print("5. Run All")
            print("0. Exit")
            
            try:
                choice = input("\nSelect demo (0-5): ").strip()
                
                if choice == '0':
                    break
                elif choice == '1':
                    demo_geometry_functions()
                elif choice == '2':
                    demo_dataset_creation(data_dir)
                elif choice == '3':
                    demo_loss_functions()
                elif choice == '4':
                    demo_configuration()
                elif choice == '5':
                    demo_geometry_functions()
                    demo_dataset_creation(data_dir)
                    demo_loss_functions()
                    demo_configuration()
                else:
                    print("Invalid choice. Please select 0-5.")
                    
            except KeyboardInterrupt:
                print("\nDemo interrupted by user.")
                break
            except Exception as e:
                print(f"Error during demo: {e}")
    
    print("\n" + "="*60)
    print("🎉 DEMO COMPLETED!")
    print("="*60)
    print("Next steps:")
    print("1. Prepare your CBCT projection data")
    print("2. Update configuration for your geometry")
    print("3. Run training: python main_cbct.py --mode cbct")
    print("4. Evaluate results: python eval_cbct.py --mode cbct")
    print("5. Run tests: python test_cbct.py")

if __name__ == '__main__':
    main()