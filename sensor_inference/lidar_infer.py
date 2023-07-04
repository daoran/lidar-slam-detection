import copy
import numpy as np
from sensor_inference.infer_base import InferBase
from sensor_inference.utils.config import cfg, cfg_from_yaml_file

def parse_config(cfg_file):
    cfg_from_yaml_file(cfg_file, cfg)
    return cfg

def build_lidar_engine(config, scn_file, rpn_file):
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

def lidar_infer(engine, input_tuple):
    return engine(input_tuple)

class LidarInfer(InferBase):
    def __init__(self, engine_start, cfg_file = None, logger = None, max_size = 3):
        super().__init__('lidarDet', engine_start, cfg_file, logger, max_size)

    def initialize(self):
        if self.cfg_file is not None:
            self.cfg = copy.deepcopy(parse_config(self.cfg_file))
        self.create_queue()

    def build_engine(self, calib):
        from sensor_inference.utils.lidar_post_process import PostProcesser
        self.engine = build_lidar_engine(self.cfg.DATA_CONFIG, self.cfg.SCN_ONNX_FILE, self.cfg.RPN_TRT_FILE)
        self.post_processer = PostProcesser(model_cfg=self.cfg.MODEL,
                                            num_class=len(self.cfg.CLASS_NAMES),
                                            class_names=self.cfg.CLASS_NAMES)

    def prepare_data(self, data_dict):
        # pop out non-relative data of lidar
        data_dict.pop('image', None)
        data_dict.pop('image_param', None)

        # seperate the data dict
        points = data_dict.pop('points', None)
        lidar_dict = {'lidar_data' : points, 'infos' : data_dict}
        return lidar_dict

    def process(self, data_dict):
        if not data_dict or not data_dict['infos']['lidar_valid']:
            return None

        points = data_dict['lidar_data']

        # cnn
        cls_preds, box_preds, label_preds = lidar_infer(self.engine, np.concatenate(list(points.values()), axis=0))

        # postprocess
        pred_dicts = self.post_processer.forward(cls_preds, box_preds, label_preds)
        data_dict['infos'].update(pred_dicts)

        result = {'lidar' : data_dict['infos']}
        return result