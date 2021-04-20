# Computer Vision-Based Guidance Assistance Concept for Plowing Using RGB-D Camera

## Setup Environment & Install Dependencies

1) Setup conda environment & install conda packages:

    `conda env create -f environment.yml`

2) Activate created environment:

    `conda activate furrow-edge-detector`

3) Install pip packages:

    `pip install pyrealsense2 tensorboard==2.4.1`

Notes about **environment.yml**:
1. `pyrealsense2` library is not needed if no frames are to be extracted from a rosbag file.
2. `pytorch` & `tensorboard` are not needed unless HED is to be trained. 
3. `pytorch` will be downloaded along with `cudatoolkit` by default. Unless a GPU is available, it is not required.

## General Information

- Most of the Python files in this repository are **not** standalone scripts. They are often imported as modules.
- Notebooks give a high-level interface to source codes to perform tasks mentioned in the paper. 
- Existing notebooks can be modified or new notebooks can be defined in **notebooks/** folder to do experiments.
- Detailed information about the usage and parameters of methods imported in notebooks can be found in the respective source code files in **src/** & **utils/** folders.

## Folders & Files

**notebooks/**
- **<span>Template Matching - Furrow Edge Detection.ipynb</span>**: Applies Template Matching method on a given dataset folder and stores the results.
- **<span>Template Matching - Artificial Road and Lane Generation.ipynb</span>**: Given a single frame, generates a new edge by using detected edge and additionally provided width information. Then, draws lane lines between the two edges.
- **<span>HED - Furrow Edge Detection.ipynb</span>**: Configures parameters for data loading, network architecture, training and logging. Then, performs training while keeping the log of the results.
- **<span>HED - Data Augmentation.ipynb</span>**: Applies random augmentation to each frame in a dataset folder and stores the results.
- **<span>Benchmark.ipynb</span>**: Compares several methods mentioned in the paper visually on the same data.

**src/**
- **<span>dataloader.py</span>**: Defines `FurrowDataset` class. This class is responsible for reading a dataset folder. It provides two methods for this purpose: (1) `get_frame_files` is the general purpose method for reading RGB, depth and edge information, (2) `__getitem__` is a method for the same purpose but specialized for the use with PyTorch.
- **<span>image_processing.py</span>**: Contains various classical computer vision methods. Notable are Otsu Thresholding, Canny Edge Detector, Template Matching.
- **<span>model.py</span>**: Defines HED architecture.
- **<span>solver.py</span>**: Implements methods for training, validating, testing the network and also logging functionality for results.

**utils/**
- **<span>extract_frames.py</span>**: Standalone script for extracting frames from a video saved in rosbag format. For detailed usage (in **utils/** folder): `./extract_frames -h`
- **<span>augment_frames.py</span>**: Responsible for random augmentation of a single frame.
- **<span>helpers.py</span>**: Contains various methods imported from several source code files and notebooks.