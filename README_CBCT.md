# Polyner CBCT - Cone Beam CT Metal Artifact Reduction

This repository extends the original Polyner implementation to support Cone Beam CT (CBCT) projection data for 3D metal artifact reduction. The modifications allow reconstruction of full 3D volumes from CBCT projections while maintaining the polychromatic neural representation approach.

![CBCT Polyner Overview](gif/fig_method.jpg)
*Figure 1: Extended Polyner model for CBCT reconstruction*

## 🆕 What's New in CBCT Version

### Key Extensions
- **3D Cone Beam Geometry**: Full support for CBCT projection geometry with source-detector trajectories
- **3D Volume Reconstruction**: Neural representation learns complete 3D volumes instead of single slices  
- **Enhanced Loss Functions**: New 3D-aware loss functions with spatial and energy smoothness constraints
- **Flexible Data Handling**: Support for both 2D sinograms and full 3D projection stacks
- **Comprehensive Evaluation**: 3D metrics including slice-by-slice analysis and volumetric assessment

### Backward Compatibility
- Original fan-beam mode fully preserved and functional
- Automatic detection and handling of both 2D and 3D data formats
- Legacy configuration files work with minimal modifications

## 📁 Extended File Structure
```
Polyner-CBCT/
│  config.json                     # Enhanced configuration with CBCT parameters
│  main_cbct.py                    # CBCT-specific training script
│  Polyner_CBCT.py                 # CBCT training implementation
│  eval_cbct.py                    # CBCT evaluation and metrics
│  test_cbct.py                    # Comprehensive testing framework
│  README_CBCT.md                  # This documentation
│  
├─ Original Files (Modified)
│  │  dataset.py                   # Extended with CBCT dataset classes
│  │  model.py                     # Added CBCT-specific loss functions
│  │  utils.py                     # Enhanced with 3D geometry functions
│  │  Polyner.py                   # Original fan-beam (preserved)
│  │  main.py                      # Original training script (preserved)
│  │  eval.py                      # Original evaluation (preserved)
│  
├─ input/
│  │  cbct_projections_x.nii       # 3D CBCT projection data (new format)
│  │  gt_volume_x.nii             # 3D ground truth volumes
│  │  mask_volume_x.nii           # 3D metal masks
│  │  ma_sinogram_x.nii           # 2D sinograms (legacy format)
│  │  ...
│  
├─ output/
│  │  cbct_polyner_x.nii          # 3D CBCT reconstructions
│  │  polyner_x.nii               # 2D fan-beam reconstructions (legacy)
│  
└─ model/
   │  cbct_model_x.pkl             # CBCT trained models
   │  model_x.pkl                  # Fan-beam trained models (legacy)
```

## ⚙️ Configuration

### CBCT Parameters
The configuration now supports additional CBCT-specific parameters:

```json
{
    "file": {
        "projection_mode": "cbct",        // "cbct" or "fan_beam"
        "SOD": 362,                       // Source-to-Object Distance
        "SDD": 652,                       // Source-to-Detector Distance  
        "h": 256, "w": 256, "d": 256,    // Volume dimensions (h×w×d)
        "detector_height": 400,           // Detector pixels (vertical)
        "detector_width": 400,            // Detector pixels (horizontal)
        "num_projections": 360,           // Number of projection angles
        ...
    },
    "network": {
        "input_dims": 3,                  // 3D coordinate input
        "n_hidden_layers": 3,             // Enhanced for 3D complexity
        ...
    }
}
```

### Geometry Specifications
- **SOD (Source-to-Object Distance)**: Distance from X-ray source to rotation center
- **SDD (Source-to-Detector Distance)**: Distance from X-ray source to detector plane
- **Detector Dimensions**: Physical detector size in pixels (height × width)
- **Volume Dimensions**: Reconstruction volume size (depth × height × width)

## 🚀 Usage

### CBCT Training
```bash
# Train on CBCT data (recommended)
python main_cbct.py --mode cbct --img_id 0

# Train all images with CBCT mode
python main_cbct.py --mode cbct

# Use custom configuration
python main_cbct.py --config custom_cbct_config.json --mode cbct
```

### Legacy Fan-Beam Training
```bash  
# Original fan-beam mode (preserved)
python main_cbct.py --mode fan_beam
python main.py  # Original script still works
```

### Data Format Requirements

#### CBCT Projection Data
- **3D Format**: (num_angles, detector_height, detector_width)
- **2D Format**: (num_angles, detector_width) - automatically extended to 3D
- **File Format**: NIfTI (.nii) files compatible with SimpleITK
- **Coordinate System**: Normalized to [-1, 1] range

#### Volume Data  
- **Ground Truth**: (depth, height, width) 3D volumes
- **Metal Masks**: (depth, height, width) binary masks (1=metal, 0=tissue)
- **Reconstruction Output**: (depth, height, width) 3D volumes

## 🔬 Evaluation

### Comprehensive Metrics
```bash
# Evaluate CBCT reconstructions
python eval_cbct.py --mode cbct

# Compare CBCT vs traditional methods
python eval_cbct.py --mode both

# Evaluate specific image
python eval_cbct.py --img_id 0 --mode cbct
```

### Available Metrics
- **PSNR/SSIM**: Standard image quality metrics
- **3D Volume Metrics**: Volumetric MSE, MAE, SNR
- **Slice-by-Slice Analysis**: Per-slice quality assessment
- **Metal Artifact Reduction**: Focused evaluation in non-metal regions

### Sample Output
```
CBCT Polyner Evaluation Results
==============================
Image ID: 0
  CBCT:
    PSNR: 35.67 dB
    SSIM: 0.9823
    MSE: 0.0027
    MAE: 0.0143
  FBP:
    PSNR: 28.34 dB  
    SSIM: 0.7856
```

## 🧪 Testing

### Comprehensive Test Suite
```bash
# Run all tests
python test_cbct.py

# Test specific components
python -m unittest test_cbct.TestCBCTGeometry
python -m unittest test_cbct.TestCBCTDataset
python -m unittest test_cbct.TestCBCTModel
```

### Test Coverage
- **Geometry Functions**: 3D ray generation, coordinate transformations
- **Dataset Handling**: CBCT data loading, legacy compatibility  
- **Model Components**: 3D loss functions, network interfaces
- **Training Workflow**: End-to-end training validation
- **Integration Tests**: Real data compatibility (when available)

## 🔄 Migration from Fan-Beam

### For Existing Users
1. **Update Configuration**: Add CBCT parameters to existing `config.json`
2. **Prepare 3D Data**: Convert 2D sinograms to 3D format or use existing CBCT data
3. **Switch Training Script**: Use `main_cbct.py` instead of `main.py`
4. **Update Evaluation**: Use `eval_cbct.py` for 3D metrics

### Data Conversion Example
```python
# Convert 2D sinogram stack to 3D CBCT format
import numpy as np
import SimpleITK as sitk

# Load multiple 2D slices
slices = []
for i in range(num_slices):
    slice_data = sitk.GetArrayFromImage(sitk.ReadImage(f'slice_{i}.nii'))
    slices.append(slice_data)

# Stack into 3D volume  
volume_3d = np.stack(slices, axis=1)  # (angles, slices, detectors)
sitk.WriteImage(sitk.GetImageFromArray(volume_3d), 'cbct_projections.nii')
```

## 📊 Performance Considerations

### Computational Requirements
- **Memory**: ~8GB+ GPU memory recommended for full 3D volumes
- **Training Time**: 2-4x longer than 2D due to 3D complexity
- **Storage**: 3D data requires significantly more disk space

### Optimization Tips
- **Batch Processing**: Use smaller batch sizes for large volumes
- **Progressive Training**: Start with smaller volumes, increase gradually
- **Mixed Precision**: Use FP16 training to reduce memory usage
- **Checkpoint Resuming**: Save frequent checkpoints for long training runs

## 🐛 Troubleshooting

### Common Issues
1. **Memory Errors**: Reduce volume size or batch size in configuration
2. **Geometry Mismatch**: Verify SOD/SDD parameters match your CBCT system
3. **Data Format**: Ensure NIfTI files have correct orientation and spacing
4. **Legacy Compatibility**: Use `projection_mode: "fan_beam"` for original data

### Debug Mode
```bash
# Enable verbose output
python main_cbct.py --mode cbct --img_id 0 --verbose

# Test with minimal configuration
python test_cbct.py --integration
```

## 📚 Technical Details

### 3D Neural Representation
- **Input**: 3D coordinates (x, y, z) ∈ [-1, 1]³
- **Output**: Multi-energy attenuation coefficients per voxel
- **Architecture**: Enhanced MLP with 3D coordinate encoding

### CBCT Forward Model
```python
# Simplified forward projection
intensity = network(ray_coords)  # (batch, samples, depth, energies)
projection = torch.exp(-voxel_size * torch.sum(intensity, dim=2))
measurement = -torch.log(torch.sum(projection * spectrum, dim=-1))
```

### Loss Function Components
- **Data Consistency**: L1 loss between predicted and measured projections  
- **Energy Smoothness**: Regularization across energy channels
- **Spatial Smoothness**: 3D spatial coherence along ray directions
- **Metal Mask Integration**: Focused constraints in non-metal regions

## 🔬 Research Applications

### Supported Scenarios
- **Clinical CBCT**: Dental, orthopedic, interventional imaging
- **Preclinical Imaging**: Small animal CBCT systems
- **Industrial CT**: Non-destructive testing with metal artifacts
- **Simulation Studies**: Synthetic CBCT data validation

### Customization Points  
- **Geometry Configuration**: Easy adaptation to different CBCT systems
- **Loss Function Weighting**: Tunable regularization parameters
- **Network Architecture**: Scalable to different volume sizes
- **Energy Spectrum**: Customizable polychromatic modeling

## 📜 Citation

If you use this CBCT extension in your research, please cite both the original Polyner paper and mention the CBCT extension:

```bibtex
@inproceedings{
wu2023unsupervised,
title={Unsupervised Polychromatic Neural Representation for {CT} Metal Artifact Reduction},
author={Qing Wu and Lixuan Chen and Ce Wang and Hongjiang Wei and S Kevin Zhou and Jingyi Yu and Yuyao Zhang},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=xx3QgKyghS}
}

% For CBCT extension, add:
@software{polyner_cbct2024,
  title={Polyner-CBCT: Cone Beam CT Extension for 3D Metal Artifact Reduction},
  year={2024},
  note={Extension of Polyner for CBCT reconstruction}
}
```

## 🔗 Additional Resources

- [Original Polyner Repository](https://github.com/original-repo-link)
- [CBCT Reconstruction Theory](https://link-to-cbct-theory)
- [ITK-SNAP Visualization](http://www.itksnap.org/pmwiki/pmwiki.php?n=Downloads.SNAP4)
- [SimpleITK Documentation](https://simpleitk.readthedocs.io/)

## 🤝 Contributing

Contributions to improve CBCT functionality are welcome! Please:

1. Fork the repository
2. Create a feature branch for CBCT enhancements  
3. Add comprehensive tests for new functionality
4. Update documentation accordingly
5. Submit pull request with detailed description

## 📄 License

This CBCT extension maintains the same license as the original Polyner implementation. Available for non-commercial research and education purposes only.

---

**Note**: This extension significantly enhances the original Polyner method for 3D CBCT reconstruction while maintaining full backward compatibility with existing fan-beam implementations.