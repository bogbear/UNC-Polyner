# ----------------------------------------------#
# Pro    : cbct
# File   : main_cbct.py
# Date   : 2024/08/25
# Author : Modified for CBCT
# Email  : 
# ----------------------------------------------#
import Polyner_CBCT
import commentjson as json
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='CBCT Polyner Training')
    parser.add_argument('--config', type=str, default='config.json', 
                      help='Configuration file path')
    parser.add_argument('--img_id', type=int, default=None,
                      help='Image ID to process (if None, processes all images 0-9)')
    parser.add_argument('--mode', type=str, default='cbct', choices=['cbct', 'fan_beam'],
                      help='Projection mode: cbct or fan_beam')
    
    args = parser.parse_args()
    
    # Load configuration
    # -----------------------
    with open(args.config) as config_file:
        config = json.load(config_file)
    
    # Override projection mode if specified
    config["file"]["projection_mode"] = args.mode
    
    # Create output directories if they don't exist
    os.makedirs(config["file"]["model_dir"], exist_ok=True)
    os.makedirs(config["file"]["out_dir"], exist_ok=True)
    
    # Print configuration summary
    print("="*50)
    print("CBCT Polyner Configuration")
    print("="*50)
    print(f"Projection mode: {config['file']['projection_mode']}")
    if args.mode == 'cbct':
        print(f"Volume size: {config['file']['h']} x {config['file']['w']} x {config['file']['d']}")
        print(f"Detector size: {config['file']['detector_height']} x {config['file']['detector_width']}")
        print(f"SOD: {config['file']['SOD']}, SDD: {config['file']['SDD']}")
        print(f"Number of projections: {config['file']['num_projections']}")
    print(f"Training epochs: {config['train']['epoch']}")
    print(f"Learning rate: {config['train']['lr']}")
    print("="*50)
    
    # Training
    # -----------------------
    if args.img_id is not None:
        # Process single image
        print(f"Processing image ID: {args.img_id}")
        Polyner_CBCT.train(img_id=args.img_id, config=config)
    else:
        # Process all images (0-9)
        print("Processing all images (0-9)")
        for i in range(10):
            print(f"\nStarting training for image {i}...")
            try:
                Polyner_CBCT.train(img_id=i, config=config)
                print(f"Completed training for image {i}")
            except Exception as e:
                print(f"Error processing image {i}: {e}")
                continue
    
    print("Training completed!")

if __name__ == '__main__':
    main()