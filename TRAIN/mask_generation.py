from ultralytics.data.annotator import auto_annotate

# Automatically detect people and segment them with high accuracy
auto_annotate(
    data="..\IMAGES\Processed_Images\Cropped\Train", 
    det_model="yolo11x.pt",  # High-accuracy detection model 
    sam_model="sam2.1_l.pt", # Your high-accuracy SAM model
    classes=[0]  # Class 0 is "person" in COCO dataset
)