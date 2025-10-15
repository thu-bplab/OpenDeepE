# OpenDeepE: Open-Source Deep-E / 2D Fully-Dense U-Net
Non-official Pytorch implementation of "Deep-E: a fully-dense neural network for improving the elevation resolution in linear-array-based photoacoustic tomography" (IEEE TMI)

## Setup

### Installation
```
git clone https://github.com/thu-bplab/OpenDeepE
cd OpenDeepE
```

### Environment
Install requirements for OpenDeepE first.
```
pip install -r requirements.txt
```

## Dataset Organization
### Directory Structure
The dataset should be organized in the following hierarchical structure:
```
dataset_root/
├── human_vessel/           # Data category
    ├── train/              # Training set
    │   ├── sparse_image/   # Sparse image data
    │   └── gt_image/       # Ground truth images
    ├── val/                # Validation set
    │   ├── sparse_image/
    │   └── gt_image/
    └── test/               # Test set
        ├── sparse_image/
        └── gt_image/
```

### File Naming Convention
All data files are in NIfTI format (.nii) with the following naming pattern:
```
{sample_index}_{slice_index}.nii
```

### Naming Explanation:
- `sample_index`: 3D data sample number (e.g., 1, 2, 3...)
- `slice_index`: 2D slice index (e.g., 90, 91, 92...)

### Examples:
- `1_90.nii` - Slice 90 from 3D sample 1
- `1_91.nii` - Slice 91 from 3D sample 1
- `1_92.nii` - Slice 92 from 3D sample 1
- `2_90.nii` - Slice 90 from 3D sample 2

### Data Description
- sparse_image: Contains input sparse image data
- gt_image: Contains corresponding ground truth image data
- The code defaults to reading NIfTI format files

### Usage Notes
Ensure that each sample has corresponding files with identical names and counts in both `sparse_image` and `gt_image` directories to maintain data alignment during training.

### Example Structure
```
human_vessel/
├── train/
│   ├── sparse_image/
│   │   ├── 1_90.nii
│   │   ├── 1_91.nii
│   │   ├── 1_92.nii
│   │   └── ...
│   └── gt_image/
│       ├── 1_90.nii
│       ├── 1_91.nii
│       ├── 1_92.nii
│       └── ...
├── val/
│   ├── sparse_image/
│   └── gt_image/
└── test/
    ├── sparse_image/
    └── gt_image/
```
This organization ensures proper data handling and compatibility with the provided codebase.


## Training
### Tips
- The recommended PyTorch version is `>=2.0`. Code is developed and tested under PyTorch `2.0.1`.
- If you encounter CUDA OOM issues, please try to reduce the `batch_size` in the training and inference configs.
- The details related to the original network are all in [Model Detail](models/FDUNet2D.py). Each processing step is explained with comments, which corresponds to Figure 1 of the original paper.

### Configuration
- Our sample training defaults to use 1 gpu with `fp32` precision.
- You may modify the configuration to fit your own environment.

### Run Training
- Please replace data related paths in the script file or `train.py` with your own paths and customize the training settings.
- An example training usage is as follows:
  ```
  # Example usage
  cd scripts
  bash train.sh
  ```

### Inference on Trained Models
- You need to modify the `$MODEL_PATH` in the testing script.
- An example inference usage is as follows:
  ```
  bash test.sh
  ```

## Acknowledgement

- We thank the authors of the [original paper](https://ieeexplore.ieee.org/abstract/document/9656766) for their great work!
- This project is supported by Beijing Academy of Artificial Intelligence by providing the computing resources.
- This project is advised by Cheng Ma.

## Citation
```
@article{zhang2021deep,
  title={Deep-E: a fully-dense neural network for improving the elevation resolution in linear-array-based photoacoustic tomography},
  author={Zhang, Huijuan and Bo, Wei and Wang, Depeng and DiSpirito, Anthony and Huang, Chuqin and Nyayapathi, Nikhila and Zheng, Emily and Vu, Tri and Gong, Yiyang and Yao, Junjie and others},
  journal={IEEE Transactions on Medical Imaging},
  volume={41},
  number={5},
  pages={1279--1288},
  year={2021},
  publisher={IEEE}
}
```
