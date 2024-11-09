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
base_dataset_path = '/content/drive/MyDrive/Data/A2/Channel2/'

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
![frame_35125](https://github.com/user-attachments/assets/71957290-7864-453a-ad18-d2fef3999ebd)![frame_37125](https://github.com/user-attachments/assets/ea48ccf0-957a-4b78-a36f-d0e3baf004d3)

![frame_34750](https://github.com/user-attachments/assets/4a4b1009-ce17-4f27-8b4a-d210be21d25b)![frame_42000](https://github.com/user-attachments/assets/fc1cb6b6-23c8-4cce-98fa-58bf1d452693)

![frame_34625](https://github.com/user-attachments/assets/0e36850a-4e02-41ab-ac22-1eb6def42c58)
![frame_34500](https://github.com/user-attachments/assets/21a8872e-c48a-4b8e-b59d-444c47831515)![frame_42125](https://github.com/user-attachments/assets/b7ad61cd-d4bf-4d63-a5b2-f26bdc513645)

![frame_34375](https://github.com/user-attachments/assets/6be4b5c1-1d5e-4748-b1ac-bdca33cdd616)
![frame_34250](https://github.com/user-attachments/assets/d567263c-6e97-45e1-9325-14ec05be3a06)
![frame_34125](https://github.com/user-attachments/assets/58ee2ed0-18fb-4754-8b92-bf9ff33ff874)![frame_43625](https://github.com/user-attachments/assets/471e5613-4300-464e-a7c2-fd51df90233f)


![frame_34000](https://github.com/user-attachments/assets/366a0cc3-717c-428b-9b28-ee85d643942b)
![frame_33875](https://github.com/user-attachments/assets/f8cc58c3-3a8f-4e67-aabb-dece2d2a7ccc)
![frame_33750](https://github.com/user-attachments/assets/8a2fad60-0252-45ac-8de5-9bf86b43bdc8)
![frame_33625](https://github.com/user-attachments/assets/6dc4ab90-c82a-455e-aff2-d28fe6ca9cdc)
![frame_33500](https://github.com/user-attachments/assets/c42ac901-df43-48a0-bdc2-6af8c9c14558)
![frame_33375](https://github.com/user-attachments/assets/004ae48f-1634-4c8e-bffa-7807c76b1b9e)
![frame_33250](https://github.com/user-attachments/assets/0d58685a-5bd6-4229-8fc4-496ad8d5b6f7)
![frame_33125](https://github.com/user-attachments/assets/0e8fcf37-3484-4798-bc68-c4e54980a01a)
![frame_33000](https://github.com/user-attachments/assets/37d82e79-de39-4bdc-baf5-b4f8933d2f1a)
![frame_32875](https://github.com/user-attachments/assets/639a063f-cf17-4cf4-9cbd-7a2bb5e1ddf5)
![frame_32750](https://github.com/user-attachments/assets/bcff1d77-dc7c-47e0-a8d8-237948cbbd9c)
![frame_32625](https://github.com/user-attachments/assets/7b4c7880-da38-4f2b-a2fe-e64fcfc842bb)
![frame_32500](https://github.com/user-attachments/assets/2d18f814-8859-4d8e-b7ee-04fe521fb707)
![frame_32375](https://github.com/user-attachments/assets/b33de0ee-51f6-489f-9fb4-6aac92eb191e)

![frame_21625](https://github.com/user-attachments/assets/2ebac900-2eee-447f-bad8-ae01ffe50364)
![frame_21500](https://github.com/user-attachments/assets/e1380bb7-f065-4089-92f6-16ded69d3253)
![frame_21375](https://github.com/user-attachments/assets/2d24e11b-a0ca-4306-bbcf-144366154488)
![frame_21250](https://github.com/user-attachments/assets/37f5cad6-be0a-43b1-9ed1-370786350be3)
![frame_21125](https://github.com/user-attachments/assets/d65aeca1-973c-4e51-a7c5-7dbcb397bfec)
![frame_21000](https://github.com/user-attachments/assets/f1c85a7b-dfb8-4c33-b5f8-57744c81639c)
![frame_20875](https://github.com/user-attachments/assets/532ce541-d85a-4d37-9fc9-50482ed1c377)
![frame_20750](https://github.com/user-attachments/assets/3fa1e322-0ae9-4b0b-8b82-d0f4be505c85)
![frame_19375](https://github.com/user-attachments/assets/a60860e0-65d2-4a3c-8b1b-691e74635e66)
![frame_18750](https://github.com/user-attachments/assets/1ccef403-9c00-438d-8c01-2bbc46897831)
![frame_18000](https://github.com/user-attachments/assets/f6e9388e-6497-4714-93d2-eeb97ecfde02)
![frame_17875](https://github.com/user-attachments/assets/81865a71-4ee3-4965-b71c-c71311be193d)![frame_43750](https://github.com/user-attachments/assets/5bbb1178-1373-4596-b5d2-5514a00bff4c)

![frame_17250](https://github.com/user-attachments/assets/8edff69f-32c1-4a75-ba44-29701bb9e01f)
![frame_17125](https://github.com/user-attachments/assets/7b5a3660-c93a-4b94-9ec2-2326d819501a)
![frame_17000](https://github.com/user-attachments/assets/0868b6bd-b134-4a15-b717-0999e3395296)
![frame_16875](https://github.com/user-attachments/assets/6225e0c9-a1e4-45b8-99da-ac6e0cea9217)
![frame_16750](https://github.com/user-attachments/assets/2fa4aa34-db00-4a90-9316-3b55cc5a71cb)
![frame_16625](https://github.com/user-attachments/assets/34d1adcc-7582-4e61-832b-ef87cdbfac3c)
![frame_16500](https://github.com/user-attachments/assets/67f798bb-0446-4950-a660-598d9c81e3c1)
![frame_16375](https://github.com/user-attachments/assets/9d63af99-2417-49b4-b32b-8ea587b1330a)
![frame_16250](https://github.com/user-attachments/assets/8e6cc351-f880-4e04-8737-5ad6cd04752f)
![frame_16125](https://github.com/user-attachments/assets/fe36ebe3-6fb7-4975-b9fd-045b46d6f7d3)
![frame_16000](https://github.com/user-attachments/assets/edda8458-3ee0-469d-89b8-49d866524d40)
![frame_15750](https://github.com/user-attachments/assets/e57c0b66-67d9-4c9d-ac5d-4033ba5d1bda)
![frame_15625](https://github.com/user-attachments/assets/7f99e80c-1b79-4417-81a6-8e3dc3e987aa)
![frame_15375](https://github.com/user-attachments/assets/aa5b6508-cc07-4759-b7c6-3916e24f16f9)
![frame_15250](https://github.com/user-attachments/assets/14768967-2627-4d72-adbb-fbcc07869dca)

![frame_34500](https://github.com/user-attachments/assets/b128e5b9-0cdf-4954-9d51-93376b3a5305)
![frame_33375](https://github.com/user-attachments/assets/ff325c91-5b0c-4f39-a670-0341d97a97be)
![frame_33250](https://github.com/user-attachments/assets/9a4f18fc-d55b-4d4d-8878-8dddad875101)
![frame_33125](https://github.com/user-attachments/assets/386cf925-e9d5-48fd-ad7e-32f67575a7a0)
![frame_33000](https://github.com/user-attachments/assets/c0e10c25-5799-47d0-af2a-4f0349fc22c8)
![frame_32875](https://github.com/user-attachments/assets/e96e6395-74b3-43c0-bcc8-e868d0fa08d6)
![frame_32750](https://github.com/user-attachments/assets/6e4e90e6-cf6c-4e47-a88c-118351ddf4c6)
![frame_32625](https://github.com/user-attachments/assets/c4cbddae-8b2d-43af-8218-552c26458987)
![frame_32500](https://github.com/user-attachments/assets/bee5945a-999b-4576-91a5-f7bc70822f0f)
![frame_32375](https://github.com/user-attachments/assets/038abbf2-8d4c-4a83-a3fb-3a4b863730c3)
![frame_17250](https://github.com/user-attachments/assets/e9aee2c8-8514-47f6-b177-ef92bb2d3efd)
![frame_17125](https://github.com/user-attachments/assets/6d14828a-c689-43c4-a242-2ab65fe2cec4)
![frame_17000](https://github.com/user-attachments/assets/9e5373b2-074f-49f5-966b-e201a80b0af3)
![frame_16875](https://github.com/user-attachments/assets/f5996331-7eeb-44a0-9c96-f6496cd042b0)
![frame_16750](https://github.com/user-attachments/assets/9d9b8907-e45e-42ff-8c49-0a3aaacce3b3)
![frame_16625](https://github.com/user-attachments/assets/16299c10-44de-4b93-882d-4fe36a500626)
![frame_16500](https://github.com/user-attachments/assets/3a5aac9b-e25c-44b6-a2be-f00ec4169c73)
![frame_16375](https://github.com/user-attachments/assets/4b589385-afd5-47a3-bcd1-43caf6ec11f9)
![frame_16250](https://github.com/user-attachments/assets/eea47741-7cd2-4e3b-9d00-277bbebf6d01)
![frame_16125](https://github.com/user-attachments/assets/8356fbaf-726a-4f3a-8e8a-1fdb70c06fde)
![frame_16000](https://github.com/user-attachments/assets/d8db5beb-0370-4b88-86c4-998e9c833d2a)
![frame_15750](https://github.com/user-attachments/assets/ea2d09db-d59b-4f7d-a631-e08962f66b08)
