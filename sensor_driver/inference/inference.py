import numpy as np

from inference_ext import inference_init as inference_init_impl
from inference_ext import inference_forward as inference_forward_impl

def inference_init(scn_file, rpn_file, voxel_size, coors_range, max_points, max_voxels, max_points_use):
    inference_init_impl(scn_file, rpn_file, voxel_size, coors_range, max_points, max_voxels, max_points_use)

def inference_forward(points):
    cls_preds, box_preds, label_preds = np.zeros((2048), dtype=np.float32), np.zeros((2048, 7), dtype=np.float32), np.zeros((2048), dtype=np.int32)
    inference_forward_impl(points, cls_preds, box_preds, label_preds)
    return cls_preds, box_preds, label_preds
