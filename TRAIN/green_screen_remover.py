"""
Simple Green Screen Removal using Range #2
This script removes green screen using the specific HSV Range #2 (41, 35, 45, 79, 255, 255)
"""
import cv2
import numpy as np
import os
import glob
from tqdm import tqdm

# Define paths directly
INPUT_DIR = r"..\IMAGES\Processed_Images\Mask_Generation\Train"
MASK_DIR = r"..\IMAGES\Processed_Images\Cropped\Train_auto_annotate_labels"
OUTPUT_DIR = r"..\IMAGES\Processed_Images\Green_Screen_Removal\Train\Range2"

def clear_directory(directory):
    """Clears all files in a directory if it exists, otherwise creates it."""
    if os.path.exists(directory):
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(directory, exist_ok=True)

def load_image_and_mask(img_path, mask_dir):
    """
    Load image and create initial mask.
    """
    # Get image filename
    img_name = os.path.basename(img_path)
    mask_path = os.path.join(mask_dir, os.path.splitext(img_name)[0] + '.txt')
    
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        return None, None, mask_path
        
    img_height, img_width = img.shape[:2]
    
    # Create initial mask from the non-black pixels in the input image
    initial_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    initial_mask[gray_img > 0] = 255
    
    # If mask file exists, use it as a reference
    if os.path.exists(mask_path):
        try:
            # Read segmentation data from annotation file
            annotation_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            with open(mask_path, 'r') as f:
                for line in f:
                    data = line.strip().split()
                    # Extract polygon points
                    polygon_points = []
                    for i in range(1, len(data), 2):
                        x = float(data[i]) * img_width
                        y = float(data[i+1]) * img_height
                        polygon_points.append([x, y])
                    
                    polygon_points = np.array(polygon_points, dtype=np.int32)
                    cv2.fillPoly(annotation_mask, [polygon_points], 255)
            
            # Combine with existing mask (use the more restrictive of the two)
            initial_mask = cv2.bitwise_and(initial_mask, annotation_mask)
        except Exception as e:
            print(f"Error processing mask file: {e}")
    
    return img, initial_mask, mask_path

def range2_green_screen_removal(img, initial_mask):
    """
    Remove green screen using Range #2 (41, 35, 45, 79, 255, 255).
    """
    # Apply initial mask to image
    masked_img = cv2.bitwise_and(img, img, mask=initial_mask)
    
    # Convert to HSV for color detection
    hsv = cv2.cvtColor(masked_img, cv2.COLOR_BGR2HSV)
    
    # Range #2 values
    lower_green = np.array([41, 35, 45])
    upper_green = np.array([79, 255, 255])
    
    # Create mask for this range
    background_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Create foreground mask (invert background and apply initial mask)
    foreground_mask = cv2.bitwise_not(background_mask)
    foreground_mask = cv2.bitwise_and(foreground_mask, initial_mask)
    
    # Clean up the mask using morphological operations
    kernel = np.ones((3, 3), np.uint8)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return foreground_mask

def main():
    """Main function to run the green screen removal using Range #2."""
    # Create output directories
    clear_directory(os.path.join(OUTPUT_DIR, "original"))
    clear_directory(os.path.join(OUTPUT_DIR, "result"))
    
    # Get all image files
    image_files = glob.glob(os.path.join(INPUT_DIR, "*.jpg")) + \
                  glob.glob(os.path.join(INPUT_DIR, "*.jpeg")) + \
                  glob.glob(os.path.join(INPUT_DIR, "*.png"))
    
    print(f"\n=== Processing {len(image_files)} images with Range #2 ===")
    
    # Process each image
    for img_path in tqdm(image_files, desc="Processing images"):
        # Get image filename
        img_name = os.path.basename(img_path)
        
        # Load image and mask
        img, initial_mask, mask_path = load_image_and_mask(img_path, MASK_DIR)
        if img is None:
            print(f"Warning: Could not process {img_name}, skipping...")
            continue
        
        # Save original
        cv2.imwrite(os.path.join(OUTPUT_DIR, "original", img_name), img)
        
        try:
            # Apply Range #2 green screen removal
            result_mask = range2_green_screen_removal(img, initial_mask)
            
            # Create final result
            result_img = cv2.bitwise_and(img, img, mask=result_mask)
            
            # Save result
            cv2.imwrite(os.path.join(OUTPUT_DIR, "result", img_name), result_img)
            
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
    
    print(f"\n=== Processing Complete! ===")
    print(f"Results saved in: {OUTPUT_DIR}")
    print("\nHSV Range #2 used:")
    print("Hue: 41 to 79")
    print("Saturation: 35 to 255")
    print("Value: 45 to 255")

if __name__ == "__main__":
    main()