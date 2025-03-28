# Body Shape Biometric System

This repository contains code for a biometric system that recognizes individuals based on their body shape, implemented for COMP 6211 Biometrics coursework.

## Overview

The system works in three main stages:
1. **Person Detection**: Uses YOLOv8 to detect and crop people from the original images
2. **Body Segmentation**: Uses SAM2 (Segment Anything Model 2) to create precise segmentations of body shapes
3. **Feature Extraction**: Extracts shape-based features from the segmentation masks for biometric recognition

## Installation

### 1. Install YOLO Dependencies

```bash
pip install ultralytics opencv-python torch torchvision
```

### 2. Install SAM2

```bash
# Clone the SAM2 repository
git clone https://github.com/facebookresearch/segment-anything-2
cd segment-anything-2
pip install -e .

# Install additional dependencies
pip install supervision pillow numpy matplotlib tqdm

# Download model checkpoints
cd checkpoints
./download_ckpts.sh

# Go back to your project directory
cd ../..
```

## Directory Structure

```
Processed_Images/
├── person_detection/       # YOLO-detected people
│   ├── train/             # Training set
│   └── test/              # Test set
├── segmentation/          # SAM2 segmentation results
│   ├── train/             # Training set segmentations
│   └── test/              # Test set segmentations
└── features/              # Extracted features
    ├── train/             # Training set features
    ├── test/              # Test set features
    └── visualizations/    # Feature visualizations
```

## Usage

### 1. Person Detection

Run the YOLO-based person detection:

```bash
python main.py --input_dir path/to/original/images --output_dir Processed_Images/person_detection
```

### 2. Body Segmentation

Run the SAM2-based body segmentation:

```bash
python run_segmentation.py --input_base Processed_Images/person_detection --output_base Processed_Images/segmentation
```

You can select different SAM2 model sizes:
- `--model_size tiny` (fastest, least accurate)
- `--model_size small` (default)
- `--model_size base` (better accuracy)
- `--model_size large` (most accurate, slowest)

### 3. Feature Extraction

Extract body shape features:

```bash
python feature_extraction.py --input_base Processed_Images/segmentation --output_base Processed_Images/features
```

## Understanding the Output

For each processed image, the system generates:
- `*_mask.png`: Binary segmentation mask
- `*_contour.npy`: Contour data for the body shape
- `*_visualization.jpg`: Visualization of the segmentation

The feature extraction produces:
- `body_features.npz`: Contains all extracted features
- Various visualization plots in the `visualizations` directory

## Notes on File Naming and Identity Codes

According to the dataset documentation, the identity code appears as the second numeric part in the filename:
- In filenames like `016z050pf.jpg`, the actual identity is `050` (not `016`)
- Each identity has two images: `pf` (profile-frontal) and `ps` (profile-side)

The feature extraction module handles this naming convention automatically.

## Extending the System

To extend the system:
1. Add more sophisticated features in `feature_extraction.py`
2. Implement classification algorithms using the extracted features
3. Evaluate recognition performance (CCR, EER) as required in the coursework

## Dependencies

- Python 3.7+
- PyTorch
- OpenCV
- Ultralytics YOLO
- SAM2 (Segment Anything Model 2)
- NumPy, Matplotlib, PIL