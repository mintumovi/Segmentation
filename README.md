# ============================
# 1. Install Necessary Libraries
# ============================

# Install PyTorch and torchvision
!pip install torch torchvision

# Install additional dependencies
!pip install pyyaml cython

# Install COCO API
!pip install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI

# Clone the Detectron2 repository and install it
!git clone https://github.com/facebookresearch/detectron2.git
%cd detectron2
!pip install -e .
%cd ..

# ============================
# 2. Import Necessary Libraries
# ============================

import cv2
import numpy as np
import os
from google.colab.patches import cv2_imshow
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from google.colab import drive

# ============================
# 3. Mount Google Drive
# ============================

drive.mount('/content/drive')

# ============================
# 4. Configure the Detectron2 Model
# ============================

# Initialize configuration
cfg = get_cfg()

# Merge configuration from model zoo
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

# Set threshold for this model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

# Use CPU for inference (Change to "cuda" if GPU is available)
cfg.MODEL.DEVICE = "cpu"

# Load pre-trained weights
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

# Initialize the predictor
predictor = DefaultPredictor(cfg)

# ============================
# 5. Define Utility Functions
# ============================

def random_color():
    """Generate a random color."""
    return [int(x) for x in np.random.choice(range(256), size=3)]

# ============================
# 6. Define Paths and Subdirectories
# ============================

# Path to the base dataset directory on Google Drive
base_dataset_path = '/content/drive/MyDrive/Data/A2/repo/'

# Get all subdirectories in the base dataset directory that start with 'Frames'
subdirectories = [
    os.path.join(base_dataset_path, d) for d in os.listdir(base_dataset_path)
    if os.path.isdir(os.path.join(base_dataset_path, d)) and d.startswith('Frames')
]

# ============================
# 7. Process Each Subdirectory
# ============================

for subdirectory in subdirectories:
    # Load all image paths in the current subdirectory
    image_files = [
        os.path.join(subdirectory, f) for f in os.listdir(subdirectory)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    # Get the subdirectory name
    subdirectory_name = os.path.basename(subdirectory)

    # Define paths to save processed frames
    processed_frames_with_humans_path = os.path.join(
        base_dataset_path, f"{subdirectory_name}WithHumans/")
    processed_frames_without_humans_path = os.path.join(
        base_dataset_path, f"{subdirectory_name}WithoutHumans/")

    # Ensure directories exist
    os.makedirs(processed_frames_with_humans_path, exist_ok=True)
    os.makedirs(processed_frames_without_humans_path, exist_ok=True)

    print(f"Processing subdirectory: {subdirectory_name} with {len(image_files)} images.")

    # Process each image in the current subdirectory
    for idx, image_file in enumerate(image_files, 1):
        # Read the image
        image = cv2.imread(image_file)

        if image is None:
            print(f"Warning: Unable to read image {image_file}. Skipping.")
            continue

        # Make predictions
        outputs = predictor(image)

        # Extract instances and move to CPU
        instances = outputs["instances"].to("cpu")

        # Filter instances to only include humans (class ID 0 in COCO)
        human_instances = instances[instances.pred_classes == 0]

        if len(human_instances) > 0:
            # Assign a distinct color to each detected person
            colors = [random_color() for _ in range(len(human_instances))]

            # Extract bounding boxes, masks, and scores
            pred_boxes = human_instances.pred_boxes.tensor.numpy()
            pred_masks = human_instances.pred_masks.numpy()
            pred_scores = human_instances.scores.numpy()  # Extract confidence scores

            for i in range(len(human_instances)):
                # Get the bounding box coordinates
                box = pred_boxes[i]
                start_point = (int(box[0]), int(box[1]))
                end_point = (int(box[2]), int(box[3]))
                color = colors[i]

                # Draw the bounding box
                cv2.rectangle(image, start_point, end_point, color, 2)

                # Add text for bounding box coordinates and confidence score
                label = f"{int(pred_scores[i] * 100)}%"  # Format score as a percentage
                cv2.putText(image, label, (start_point[0], start_point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

                # Optionally, add the mask
                mask = pred_masks[i]
                colored_mask = np.zeros_like(image)
                colored_mask[:, :] = color
                # Blend the mask with the image
                image = np.where(mask[:, :, np.newaxis], cv2.addWeighted(image, 1.0, colored_mask, 0.5, 0), image)

            # Save the image to the folder for frames with humans
            save_path = os.path.join(
                processed_frames_with_humans_path, os.path.basename(image_file))
        else:
            # Save the image to the folder for frames without humans
            save_path = os.path.join(
                processed_frames_without_humans_path, os.path.basename(image_file))

        # Save the processed image
        cv2.imwrite(save_path, image)

        # (Optional) Display the result for the first few images
        if idx <= 5:
            cv2_imshow(image)

    print(f"Completed processing subdirectory: {subdirectory_name}\n")

print("All subdirectories have been processed.")


