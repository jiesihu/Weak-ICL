import torch
from torch.nn import functional as F
import numpy as np
import pytorch_lightning as pl

from scipy import ndimage as ndi
from skimage.morphology import ball
from scipy.ndimage import distance_transform_edt

from .models.neuroverse3D import Neuroverse3D
from utils.pairwise_measures import BinaryPairwiseMeasures as PM

class LightningModel(pl.LightningModule):
    """
    We use pytorch lightning to organize our model code
    """

    def __init__(self, hparams):
        super().__init__()
        # save hparams / load hparams
        self.save_hyperparameters(hparams)
        self.automatic_optimization = False
        
        # build model
        self.net = Neuroverse3D(in_channels = 1, 
                    out_channels = 1, 
                    stages = len(self.hparams.nb_inner_channels), 
                    dim = 2 if self.hparams.data_slice_only else 3,
                    inner_channels = self.hparams.nb_inner_channels,
                    conv_layers_per_stage = self.hparams.nb_conv_layers_per_stage)
        
        
    def forward(self, target_in, context_in, context_out, gs= 3):
        # data normalization
#         target_in = self.normalize_3d_volume(target_in)
#         context_in = self.normalize_3d_volume(context_in)
#         context_out = self.normalize_3d_volume(context_out)
        
        # run network
        y_pred = self.net(target_in.to(self.device), context_in, context_out, l = gs)
        return y_pred
    
    def forward_bbox(self, target_in, context_in, bbox, gs = 1):
        
        context_out = self.create_bbox_masks(bbox)
        
        mask = self.forward(target_in, context_in, context_out, gs = gs)
        
        return mask
    
    def forward_point(self, target_in, context_in, point, gs = 1):
        
        context_out = self.place_spheres_at_points_for_Context(point)
        
        mask = self.forward(target_in, context_in, context_out, gs = gs)
        
        return mask
        
    def normalize_3d_volume(self, target_in: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Normalizes a batch of 3D volumes (shape: [..., D, H, W]) to the range [0, 1] independently for each sample.

        Args:
            target_in (torch.Tensor): Input tensor of shape [..., 1, D, H, W], where N is the batch size,
                                      and (D, H, W) are the depth, height, and width of the 3D volume.
            eps (float): A small value to prevent division by zero. Default is 1e-8.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as the input, where each sample is scaled to [0, 1].
        """
        # Ensure the input tensor is of type float32 to avoid integer division issues
        if target_in.dtype != torch.float32:
            target_in = target_in.to(torch.float32)

        # Compute the minimum and maximum values for each sample independently
        # Input shape: [N, 1, D, H, W] → Output shape: [N, 1, 1, 1, 1]
        min_vals = torch.amin(target_in, dim=(-3, -2, -1), keepdim=True)
        max_vals = torch.amax(target_in, dim=(-3, -2, -1), keepdim=True)

        # Compute the dynamic range and prevent division by zero
        dynamic_range = max_vals - min_vals
        dynamic_range[dynamic_range < eps] = eps  # Replace small ranges with eps to avoid division by zero

        # Normalize the input tensor to the range [0, 1]
        normalized = (target_in - min_vals) / dynamic_range

        return normalized


    def _expand_slice(self, coords, dim_size):
        """
        Expands a coordinate slice of thickness 1 to a thickness of 3, handling boundaries.

        Args:
            coords (list or tuple): The coordinate pair [start, end].
            dim_size (int): The maximum size of the dimension (e.g., 128).

        Returns:
            list: The new, possibly expanded, coordinate pair.
        """
        start, end = coords
        if end - start == 1:
            # This is a slice of thickness 1 at index `start`
            slice_index = start
            if slice_index == 0:
                # At the lower boundary, expand to [0, 3]
                return [0, 3]
            elif slice_index == dim_size - 1:
                # At the upper boundary, expand to [dim_size - 3, dim_size]
                return [dim_size - 3, dim_size]
            else:
                # Centered expansion
                return [slice_index - 1, slice_index + 2]
        else:
            # If thickness is not 1, return original coordinates
            return [start, end]

    def create_bbox_masks(self, bbox_coordinates, volume_dim=(128, 128, 128), device='cpu', expand_slices_to_3=True):
        """
        Generates a stacked binary mask tensor from a list of bounding box coordinates.
        Optionally expands axes with a thickness of 1 to a thickness of 3.

        Args:
            bbox_coordinates (list): A list of bounding box coordinates.
                Each element is formatted as [[x_start, x_end], [y_start, y_end], [z_start, z_end]].
                The coordinate range is exclusive of the end index, e.g., [30, 80] selects indices from 30 up to 79.
            volume_dim (tuple, optional): The dimensions of the 3D volume (D, H, W). Defaults to (128, 128, 128).
            device (str, optional): The computation device ('cpu' or 'cuda'). Defaults to 'cpu'.
            expand_slices_to_3 (bool, optional): If True, expands any axis with thickness 1 to thickness 3. Defaults to False.

        Returns:
            torch.Tensor: The final tensor with a shape of [1, n, 1, D, H, W],
                          where n is the number of bounding boxes.
        """
        D, H, W = volume_dim
        individual_masks = []

        for i, bbox in enumerate(bbox_coordinates):
            # Create a zero tensor for the current bbox
            mask = torch.zeros((1, 1, 1, D, H, W), dtype=torch.uint8, device=device)

            # Extract original coordinates
            x_coords, y_coords, z_coords = bbox

            # If expansion is enabled, apply it to each axis
            if expand_slices_to_3:
                final_x_coords = self._expand_slice(x_coords, W)
                final_y_coords = self._expand_slice(y_coords, H)
                final_z_coords = self._expand_slice(z_coords, D)
    #             if final_x_coords != x_coords or final_y_coords != y_coords or final_z_coords != z_coords:
    #                  print(f"  - Bbox {i}: Original {bbox} -> Expanded to [X:{final_x_coords}, Y:{final_y_coords}, Z:{final_z_coords}]")
            else:
                final_x_coords, final_y_coords, final_z_coords = x_coords, y_coords, z_coords

            # Get the final start and end points for slicing
            x_start, x_end = final_x_coords
            y_start, y_end = final_y_coords
            z_start, z_end = final_z_coords

            # Fill the region defined by the final coordinates with a value of 1
            mask[0, 0, 0, x_start:x_end, y_start:y_end, z_start:z_end] = 1

            individual_masks.append(mask)

        if not individual_masks:
            print("Input list is empty, returning an empty tensor.")
            return torch.empty((1, 0, 1, D, H, W), dtype=torch.uint8, device=device)

        final_tensor = torch.cat(individual_masks, dim=1)

        return final_tensor

    def sphere_with_normalized_edt(self, radius, ndim = 3):
        """
        生成一个半径为 radius 的球体（或圆盘），并通过 EDT 得到soft mask（中心=1, 边界=0）。
        """
        if ndim == 3:
            strel = ball(radius).astype(np.uint8)
        elif ndim == 2:
            from skimage.morphology import disk
            strel = disk(radius).astype(np.uint8)
        else:
            raise ValueError("ndim must be 2 or 3")

        edt = distance_transform_edt(strel)
        edt = edt / edt.max()  # normalize
        return torch.from_numpy(edt.astype(np.float32))


    def place_spheres_at_points(self, 
        points,
        radius = 6,
        mask_shape = (128,128,128),
        device = 'cpu'
    ):
        """
        在一个给定的3D空间中，根据指定的坐标点列表放置半径为radius的球体。

        Args:
            mask_shape (tuple): 输出掩码的形状，例如 (H, W, D) 或 (128, 128, 128)。
            points (list[tuple[int, int, int]]): 一个包含3D坐标元组的列表，格式为 [(y1, x1, z1), (y2, x2, z2), ...]。
            radius (int): 要放置的球体的半径。
            device (str): 计算设备 ('cpu' or 'cuda')。

        Returns:
            torch.Tensor: 一个新的3D张量，其中在指定位置放置了球体。
        """
        if len(mask_shape) != 3:
            raise ValueError("Expect 3D shape, e.g., (H, W, D)")

        H, W, D = mask_shape
        mask_out = torch.zeros(mask_shape, dtype=torch.float32, device=device)

        if not points:
            print("Warning: points list is empty. Returning a zero tensor.")
            return mask_out

        # 1. 只创建一次球体模板，提高效率
        sphere = self.sphere_with_normalized_edt(radius, ndim=3).to(device)
        sz = sphere.shape

        # 2. 遍历所有指定的点
        for center in points:
            if len(center) != 3:
                print(f"Skipping invalid point format: {center}. Expected (y, x, z).")
                continue

            y, x, z = center

            # 3. 计算球体的边界框 (bounding box)
            # Bbox aabb
            zmin, zmax = z - sz[2]//2, z + sz[2]//2 + 1
            ymin, ymax = y - sz[0]//2, y + sz[0]//2 + 1
            xmin, xmax = x - sz[1]//2, x + sz[1]//2 + 1

            # 4. 裁剪边界框，确保它不会超出 mask_out 的范围
            zmin_cl, zmax_cl = max(zmin, 0), min(zmax, D)
            ymin_cl, ymax_cl = max(ymin, 0), min(ymax, H)
            xmin_cl, xmax_cl = max(xmin, 0), min(xmax, W)

            # 如果裁剪后的区域无效（例如，球体完全在掩码外部），则跳过
            if (zmin_cl >= zmax_cl) or (ymin_cl >= ymax_cl) or (xmin_cl >= xmax_cl):
                continue

            # 5. 将裁剪后的球体部分放置到输出掩码中
            # 使用 torch.maximum 可以正确处理球体之间的重叠区域
            mask_out[ymin_cl:ymax_cl, xmin_cl:xmax_cl, zmin_cl:zmax_cl] = torch.maximum(
                mask_out[ymin_cl:ymax_cl, xmin_cl:xmax_cl, zmin_cl:zmax_cl],
                sphere[
                    (ymin_cl - ymin):(ymax_cl - ymin),
                    (xmin_cl - xmin):(xmax_cl - xmin),
                    (zmin_cl - zmin):(zmax_cl - zmin),
                ]
            )
        return mask_out

    def place_spheres_at_points_for_Context(self, 
            POINT_COORDINATES_CONTEXT,
            radius= 6,
            mask_shape = (128,128,128),
            device= 'cpu',
        ):

        context_out = []
        for points in POINT_COORDINATES_CONTEXT:
            _context_out = self.place_spheres_at_points(points, radius = radius, mask_shape = mask_shape, device = device)
            context_out.append(_context_out[None, None, None, :])

        context_out = torch.cat(context_out, dim=1)
        return context_out