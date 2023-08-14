import copy
import numpy as np
from sensor_inference.infer_base import InferBase
from sensor_inference.utils.config import cfg, cfg_from_yaml_file

def parse_config(cfg_file):
    cfg_from_yaml_file(cfg_file, cfg)
    return cfg

def build_engine(config, scn_file, rpn_file):
    from sensor_driver.inference.inference import inference_init, inference_forward
    inference_init(
        scn_file       = scn_file,
        rpn_file       = rpn_file,
        voxel_size     = config.VOXELIZATION.VOXEL_SIZE,
        coors_range    = np.array(config.POINT_CLOUD_RANGE, dtype=np.float32),
        max_points     = config.VOXELIZATION.MAX_POINTS_PER_VOXEL,
        max_voxels     = config.VOXELIZATION.MAX_NUMBER_OF_VOXELS['test'],
        max_points_use = config.VOXELIZATION.MAX_POINTS,
    )

    def engine(points):
        return inference_forward(points)

    return engine

class ObjectInfer(InferBase):
    def __init__(self, engine_start, cfg_file = None, logger = None, max_size = 3):
        super().__init__('Object', engine_start, cfg_file, logger, max_size)

    def initialize(self):
        self.cfg = copy.deepcopy(parse_config(self.cfg_file))
        self.create_queue()

    def build_engine(self, calib):
        from sensor_inference.utils.object_post_process import PostProcesser
        self.engine = build_engine(self.cfg.DATA_CONFIG, self.cfg.SCN_ONNX_FILE, self.cfg.RPN_TRT_FILE)
        self.post_processer = PostProcesser(model_cfg=self.cfg.MODEL,
                                            num_class=len(self.cfg.CLASS_NAMES),
                                            class_names=self.cfg.CLASS_NAMES)

    def prepare_data(self, data_dict):
        # pop out non-relative data of lidar
        data_dict.pop('image', None)
        data_dict.pop('image_param', None)

        # seperate the data dict
        points = data_dict.pop('points', None)
        return {'lidar_data' : points, 'infos' : data_dict}

    def process(self, data_dict):
        if not data_dict:
            return None

        points = np.concatenate(list(data_dict['lidar_data'].values()), axis=0) if data_dict['infos']['lidar_valid'] else np.zeros((1, 4), dtype=np.float32)

        # cnn
        cls_preds, box_preds, label_preds = self.engine(points)

        # postprocess
        pred_dicts = self.post_processer.forward(cls_preds, box_preds, label_preds)
        data_dict['infos'].update(pred_dicts)

        result = {'object' : data_dict['infos']}
        return result