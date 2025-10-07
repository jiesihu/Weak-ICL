# Efficient Universal Models for Medical Image Segmentation via Weakly Supervised In-Context Learning

This repository contains the official PyTorch implementation and pre-trained models for the paper "Efficient Universal Models for Medical Image Segmentation via Weakly Supervised In-Context Learning" (WS-ICL).

---

## Installation
### Environment Setup

Make sure you have Python and PyTorch installed. All required libraries are listed in the `requirements.txt` file. You can install them using:

```bash
pip install -r requirements.txt
```

### Checkpoints
Download the pre-trained model checkpoints from the following link:
- - Link: https://pan.baidu.com/s/1ybeS5Bh_-0jJm3QghjCRjA
- - Extraction Code: g6mh

After downloading, place the `.ckpt` files into a `Checkpoints/` directory in your project root.

## Usage
We provide two separate models, one for bounding box prompts and one for point prompts. The input image size is fixed at `[128, 128, 128]`.

### Bounding Box Prompt
This example demonstrates how to use the model with bounding box prompts for the context images.
```python
import torch
from torch.nn import functional as F
import os

# 1. Load the pre-trained model for bounding box prompts
from weak_ICL.lightning_model import LightningModel
checkpoint_path = "Checkpoints/weak_ICL_bbox.ckpt"
model = LightningModel.load_from_checkpoint(checkpoint_path, map_location=torch.device("cuda:0"))
model.eval() # Set model to evaluation mode

# 2. Generate dummy input data
# Target image shape: [batch, channels, depth, height, width]
target_in = torch.rand([1, 1, 128, 128, 128])
# Context images shape: [batch, num_context_images, channels, depth, height, width]
context_in = torch.rand([1, 4, 1, 128, 128, 128])

# 3. Define bounding box coordinates for each context image
# Format: [[[X_start, X_end], [Y_start, Y_end], [Z_start, Z_end]], ...]
# This example provides one 2D box on a specific slice for each of the 4 context images.
BBOX_COORDINATES_CONTEXT = [
    [[30, 80], [40, 100], [64, 65]], # For context image 1
    [[30, 80], [20, 40], [64, 65]],  # For context image 2
    [[80, 100], [40, 100], [64, 65]],# For context image 3
    [[80, 100], [20, 40], [64, 65]], # For context image 4
]

# 4. Normalize data (important step)
target_in = model.normalize_3d_volume(target_in)
context_in = model.normalize_3d_volume(context_in)

# 5. Perform inference
with torch.no_grad():
    mask = model.forward_bbox(target_in, context_in, BBOX_COORDINATES_CONTEXT)

# The output 'mask' will be the predicted segmentation for the target_in
print(mask.shape)

```

### Point Prompt
This example demonstrates how to use the model with point prompts for the context images. You can provide one or more points for each context image.
```python
import torch
from torch.nn import functional as F
import os

# 1. Load the pre-trained model for point prompts
from weak_ICL.lightning_model import LightningModel
checkpoint_path = "Checkpoints/weak_ICL_point.ckpt"
model = LightningModel.load_from_checkpoint(checkpoint_path, map_location=torch.device("cuda:0"))
model.eval() # Set model to evaluation mode

# 2. Generate dummy input data
target_in = torch.rand([1, 1, 128, 128, 128])
context_in = torch.rand([1, 4, 1, 128, 128, 128])

# 3. Define point coordinates for each context image
# Format: A list of lists, where each inner list contains points for one context image.
# Each point is a tuple/list of (x, y, z).
POINT_COORDINATES_CONTEXT = [
    [[44, 44, 64]],                      # 1 point for context image 1
    [[44, 44, 64], [44, 64, 64]],        # 2 points for context image 2
    [[64, 44, 64]],                      # 1 point for context image 3
    [[44, 64, 64], [64, 44, 64]],        # 2 points for context image 4
]

# 4. Normalize data
target_in = model.normalize_3d_volume(target_in)
context_in = model.normalize_3d_volume(context_in)

# 5. Perform inference
with torch.no_grad():
    mask = model.forward_point(target_in, context_in, POINT_COORDINATES_CONTEXT)

# The output 'mask' will be the predicted segmentation for the target_in
print(mask.shape)
```


## Citation
If you find this work useful in your research, please consider citing our paper: **Efficient Universal Models for Medical Image Segmentation via Weakly Supervised In-Context Learning**


## Acknowledgements
This repository was modified from [Neuroverse3D](https://github.com/jiesihu/Neuroverse3D).
