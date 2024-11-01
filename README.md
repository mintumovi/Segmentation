# Segmentation
# Install necessary libraries
!pip install torch torchvision
!pip install pyyaml cython
!pip install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI

# Clone the Detectron2 repository and install it
!git clone https://github.com/facebookresearch/detectron2.git
%cd detectron2
!pip install -e .

