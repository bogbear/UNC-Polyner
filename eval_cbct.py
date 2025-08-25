# ----------------------------------------------#
# Pro    : cbct  
# File   : eval_cbct.py
# Date   : 2024/08/25
# Author : Modified for CBCT evaluation
# Email  : 
# ----------------------------------------------#
import utils
import numpy as np
import SimpleITK as sitk
import argparse
import os

def evaluate_3d_volume(pred_volume, gt_volume, mask_volume=None):
    """
    Evaluate 3D volume reconstruction quality
    
    Args:
        pred_volume: Predicted 3D volume
        gt_volume: Ground truth 3D volume  
        mask_volume: Optional mask for evaluation region
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    if mask_volume is not None:
        # Apply mask to focus evaluation on non-metal regions
        pred_masked = pred_volume[mask_volume > 0]
        gt_masked = gt_volume[mask_volume > 0]
    else:
        pred_masked = pred_volume.flatten()
        gt_masked = gt_volume.flatten()
    
    # Calculate metrics
    psnr_val = utils.psnr(pred_masked, gt_masked)
    ssim_val = utils.ssim(pred_volume, gt_volume)
    
    # Additional 3D metrics
    mse = np.mean((pred_masked - gt_masked) ** 2)
    mae = np.mean(np.abs(pred_masked - gt_masked))
    
    # Signal-to-noise ratio
    signal_power = np.mean(gt_masked ** 2)
    noise_power = mse
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else np.inf
    
    return {
        'PSNR': psnr_val,
        'SSIM': ssim_val, 
        'MSE': mse,
        'MAE': mae,
        'SNR': snr
    }

def evaluate_slice_by_slice(pred_volume, gt_volume, mask_volume=None):
    """
    Evaluate volume slice by slice and return statistics
    """
    num_slices = pred_volume.shape[0]
    slice_metrics = []
    
    for i in range(num_slices):
        pred_slice = pred_volume[i]
        gt_slice = gt_volume[i]
        mask_slice = mask_volume[i] if mask_volume is not None else None
        
        if mask_slice is not None:
            pred_masked = pred_slice[mask_slice > 0]
            gt_masked = gt_slice[mask_slice > 0]
        else:
            pred_masked = pred_slice
            gt_masked = gt_slice
        
        if len(pred_masked) > 0:  # Only evaluate if there are pixels to evaluate
            psnr_val = utils.psnr(pred_masked, gt_masked)
            ssim_val = utils.ssim(pred_slice, gt_slice)
            slice_metrics.append({'slice': i, 'PSNR': psnr_val, 'SSIM': ssim_val})
    
    return slice_metrics

def main():
    parser = argparse.ArgumentParser(description='CBCT Polyner Evaluation')
    parser.add_argument('--input_dir', type=str, default='./input',
                      help='Input directory containing ground truth data')
    parser.add_argument('--output_dir', type=str, default='./output', 
                      help='Output directory containing reconstructions')
    parser.add_argument('--results_file', type=str, default='cbct_evaluation_results.txt',
                      help='File to save evaluation results')
    parser.add_argument('--img_id', type=int, default=None,
                      help='Specific image ID to evaluate (if None, evaluates all)')
    parser.add_argument('--mode', type=str, default='cbct', choices=['cbct', 'fan_beam', 'both'],
                      help='Evaluation mode')
    
    args = parser.parse_args()
    
    # Determine which images to evaluate
    if args.img_id is not None:
        img_ids = [args.img_id]
    else:
        img_ids = list(range(10))  # Evaluate images 0-9
    
    all_results = []
    
    print("="*60)
    print("CBCT Polyner Evaluation")
    print("="*60)
    
    for img_id in img_ids:
        print(f"\nEvaluating image {img_id}...")
        
        # File paths
        gt_path = f'{args.input_dir}/gt_{img_id}.nii'
        mask_path = f'{args.input_dir}/mask_{img_id}.nii'
        
        if args.mode in ['cbct', 'both']:
            cbct_pred_path = f'{args.output_dir}/cbct_polyner_{img_id}.nii'
            
        if args.mode in ['fan_beam', 'both']:
            fbp_path = f'{args.input_dir}/ma_{img_id}.nii'  # FBP reconstruction
            polyner_path = f'{args.output_dir}/polyner_{img_id}.nii'  # Original Polyner
        
        try:
            # Load ground truth
            if os.path.exists(gt_path):
                gt_volume = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))
                if len(gt_volume.shape) == 2:
                    gt_volume = gt_volume[np.newaxis, :, :]  # Add depth dimension
            else:
                print(f"Warning: Ground truth not found: {gt_path}")
                continue
            
            # Load mask if available
            mask_volume = None
            if os.path.exists(mask_path):
                mask_volume = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
                if len(mask_volume.shape) == 2:
                    mask_volume = mask_volume[np.newaxis, :, :]  # Add depth dimension
                mask_volume = (mask_volume == 0).astype(np.float32)  # Invert mask for evaluation regions
            
            results = {'img_id': img_id}
            
            # Evaluate CBCT reconstruction
            if args.mode in ['cbct', 'both'] and os.path.exists(cbct_pred_path):
                cbct_volume = sitk.GetArrayFromImage(sitk.ReadImage(cbct_pred_path))
                
                # Ensure volumes have same shape
                if cbct_volume.shape != gt_volume.shape:
                    print(f"Warning: Shape mismatch for CBCT volume {img_id}")
                    print(f"  Predicted: {cbct_volume.shape}, GT: {gt_volume.shape}")
                    # Resize if needed (simple approach)
                    min_shape = tuple(min(p, g) for p, g in zip(cbct_volume.shape, gt_volume.shape))
                    cbct_volume = cbct_volume[:min_shape[0], :min_shape[1], :min_shape[2]]
                    gt_volume_cbct = gt_volume[:min_shape[0], :min_shape[1], :min_shape[2]]
                    mask_volume_cbct = mask_volume[:min_shape[0], :min_shape[1], :min_shape[2]] if mask_volume is not None else None
                else:
                    gt_volume_cbct = gt_volume
                    mask_volume_cbct = mask_volume
                
                cbct_metrics = evaluate_3d_volume(cbct_volume, gt_volume_cbct, mask_volume_cbct)
                results['CBCT'] = cbct_metrics
                print(f"  CBCT - PSNR: {cbct_metrics['PSNR']:.2f}, SSIM: {cbct_metrics['SSIM']:.4f}")
            
            # Evaluate traditional methods for comparison
            if args.mode in ['fan_beam', 'both']:
                # FBP evaluation
                if os.path.exists(fbp_path):
                    fbp_volume = sitk.GetArrayFromImage(sitk.ReadImage(fbp_path))
                    if len(fbp_volume.shape) == 2:
                        fbp_volume = fbp_volume[np.newaxis, :, :]
                    fbp_metrics = evaluate_3d_volume(fbp_volume, gt_volume, mask_volume)
                    results['FBP'] = fbp_metrics
                    print(f"  FBP - PSNR: {fbp_metrics['PSNR']:.2f}, SSIM: {fbp_metrics['SSIM']:.4f}")
                
                # Original Polyner evaluation
                if os.path.exists(polyner_path):
                    polyner_volume = sitk.GetArrayFromImage(sitk.ReadImage(polyner_path))
                    if len(polyner_volume.shape) == 2:
                        polyner_volume = polyner_volume[np.newaxis, :, :]
                    polyner_metrics = evaluate_3d_volume(polyner_volume, gt_volume, mask_volume)
                    results['Polyner_2D'] = polyner_metrics
                    print(f"  Polyner 2D - PSNR: {polyner_metrics['PSNR']:.2f}, SSIM: {polyner_metrics['SSIM']:.4f}")
            
            all_results.append(results)
            
        except Exception as e:
            print(f"Error evaluating image {img_id}: {e}")
            continue
    
    # Calculate and display summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    if all_results:
        methods = set()
        for result in all_results:
            methods.update(result.keys())
        methods.discard('img_id')  # Remove img_id from methods
        
        for method in methods:
            method_results = [r[method] for r in all_results if method in r]
            if method_results:
                psnr_values = [m['PSNR'] for m in method_results]
                ssim_values = [m['SSIM'] for m in method_results]
                
                psnr_mean = np.mean(psnr_values)
                psnr_std = np.std(psnr_values)
                ssim_mean = np.mean(ssim_values)
                ssim_std = np.std(ssim_values)
                
                print(f"{method}:")
                print(f"  PSNR: {psnr_mean:.2f} ± {psnr_std:.2f}")
                print(f"  SSIM: {ssim_mean:.4f} ± {ssim_std:.4f}")
                print()
    
    # Save detailed results to file
    with open(args.results_file, 'w') as f:
        f.write("CBCT Polyner Evaluation Results\n")
        f.write("="*60 + "\n\n")
        
        for result in all_results:
            f.write(f"Image ID: {result['img_id']}\n")
            for method, metrics in result.items():
                if method != 'img_id':
                    f.write(f"  {method}:\n")
                    for metric, value in metrics.items():
                        f.write(f"    {metric}: {value:.4f}\n")
            f.write("\n")
        
        # Summary statistics
        f.write("SUMMARY STATISTICS\n")
        f.write("="*30 + "\n")
        
        methods = set()
        for result in all_results:
            methods.update(result.keys())
        methods.discard('img_id')
        
        for method in methods:
            method_results = [r[method] for r in all_results if method in r]
            if method_results:
                psnr_values = [m['PSNR'] for m in method_results]
                ssim_values = [m['SSIM'] for m in method_results]
                
                f.write(f"\n{method}:\n")
                f.write(f"  PSNR: {np.mean(psnr_values):.2f} ± {np.std(psnr_values):.2f}\n")
                f.write(f"  SSIM: {np.mean(ssim_values):.4f} ± {np.std(ssim_values):.4f}\n")
    
    print(f"Detailed results saved to: {args.results_file}")

if __name__ == '__main__':
    main()