"""
Silhouette generation from segmentation masks.
This script applies segmentation masks to cropped images to create silhouettes.
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path

def clear_directory(directory):
    """
    Clears all files in a directory if it exists, otherwise creates it.
    
    Args:
        directory (str): Path to the directory to clear.
    """
    if os.path.exists(directory):
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(directory, exist_ok=True)

def apply_segmentation_mask(img_path, mask_path, output_path, save_mask=False):
    """
    Apply segmentation mask to an image and save the result.
    
    Args:
        img_path (str): Path to the input image
        mask_path (str): Path to the segmentation mask txt file
        output_path (str): Path to save the masked image
        save_mask (bool): Whether to save the binary mask as a separate file
    
    Returns:
        bool: True if processing was successful, False otherwise
    """
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error loading image: {img_path}")
        return False
    
    img_height, img_width = img.shape[:2]
    
    # Create empty mask
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    # Check if mask file exists
    if not os.path.exists(mask_path):
        print(f"Mask file not found: {mask_path}")
        return False
    
    try:
        # Read segmentation data from txt file
        with open(mask_path, 'r') as f:
            for line in f:
                data = line.strip().split()
                class_id = int(data[0])  # Should be 0 for person
                
                # Extract polygon points
                polygon_points = []
                for i in range(1, len(data), 2):
                    # Convert normalized coordinates back to pixel values
                    x = float(data[i]) * img_width
                    y = float(data[i+1]) * img_height
                    polygon_points.append([x, y])
                
                # Convert to numpy array of integer points
                polygon_points = np.array(polygon_points, dtype=np.int32)
                
                # Draw filled polygon on mask
                cv2.fillPoly(mask, [polygon_points], 255)
        
        # Apply mask to isolate the person
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        
        # Save the masked image
        cv2.imwrite(output_path, masked_img)
        
        # Optionally save the binary mask
        if save_mask:
            mask_output_path = output_path.replace('.jpg', '_mask.png')
            cv2.imwrite(mask_output_path, mask)
        
        return True
    
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return False

def process_directory(input_dir, mask_dir, output_dir, save_masks=False):
    """
    Process all images in a directory to apply segmentation masks.
    
    Args:
        input_dir (str): Directory containing input cropped images
        mask_dir (str): Directory containing segmentation mask txt files
        output_dir (str): Directory to save masked images
        save_masks (bool): Whether to save binary masks as separate files
    """
    # Ensure output directory exists
    clear_directory(output_dir)
    
    # Get all image files
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"\n=== Processing {len(files)} images for silhouette generation ===")
    
    success_count = 0
    for idx, file in enumerate(files):
        print(f"Processing {idx+1}/{len(files)}: {file}")
        
        img_path = os.path.join(input_dir, file)
        mask_path = os.path.join(mask_dir, file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))
        output_path = os.path.join(output_dir, file)
        
        if apply_segmentation_mask(img_path, mask_path, output_path, save_masks):
            success_count += 1
            print(f"Successfully created silhouette: {output_path}")
        else:
            # Copy original if segmentation fails
            print(f"Segmentation failed, copying original: {file}")
            try:
                cv2.imwrite(output_path, cv2.imread(img_path))
            except:
                print(f"Could not copy original image: {img_path}")
    
    print(f"\n=== Silhouette Generation Complete ===")
    print(f"Successfully processed {success_count}/{len(files)} images")
    print(f"Results saved to: {output_dir}")

def main():
    """Main function to run the silhouette generation pipeline"""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Silhouette Generation Pipeline')
    parser.add_argument('--input_dir', default=r"..\IMAGES\Processed_Images\Cropped\Train", 
                      help='Path to input cropped images')
    parser.add_argument('--mask_dir', default=r"..\IMAGES\Processed_Images\Cropped\Train_auto_annotate_labels", 
                      help='Path to segmentation mask txt files')
    parser.add_argument('--output_dir', default=r"..\IMAGES\Processed_Images\Mask_Generation\Train", 
                      help='Path to store silhouettes')
    parser.add_argument('--save_masks', action='store_true',
                      help='Save binary masks as separate files')
    
    args = parser.parse_args()
    
    # Process all images
    process_directory(args.input_dir, args.mask_dir, args.output_dir, args.save_masks)

if __name__ == "__main__":
    main()