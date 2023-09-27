import os
import copy
import numpy as np
from easydict import EasyDict
from sensor_inference.infer_base import InferBase
from sensor_inference.utils.config import cfg_from_yaml_file
from util.image_util import cvt_image

def parse_config(cfg_file):
    cfg = EasyDict()
    cfg_from_yaml_file(cfg_file, cfg)
    return cfg

def build_engine(trt_file):
    from sensor_inference.trt_engine.runtime_engine import RuntimeBackend as backend
    engine = backend(trt_path=trt_file)
    return engine

class TrafficLightInfer(InferBase):
    def __init__(self, engine_start, cfg_file = None, logger = None, max_size = 3):
        super().__init__('TrafficLight', engine_start, cfg_file, logger, max_size)

    def initialize(self):
        self.cfg_file = self.cfg_file if os.path.exists(self.cfg_file) else 'sensor_inference/cfgs/detection_trafficlight.yaml'
        self.cfg = copy.deepcopy(parse_config(self.cfg_file))
        self.create_queue()

    def build_engine(self, calib):
        from sensor_inference.utils.trafficlight_post_process import PostProcesser
        self.engine = build_engine(self.cfg.TRT_FILE)
        self.post_processer = PostProcesser(model_cfg=self.cfg.MODEL, 
                                            num_class=len(self.cfg.CLASS_NAMES), 
                                            class_names=self.cfg.CLASS_NAMES)

    def prepare_data(self, data_dict):
        # pop out non-relative data of image
        data_dict.pop('points', None)

        # seperate the data dict
        images = data_dict.pop('image', None)
        return {'image_data' : images, 'infos' : data_dict}

    def process(self, data_dict):
        if not data_dict:
            return None

        if not data_dict['infos']['image_valid']:
            return {'trafficlight' : None}

        names, images = [], []
        for name, image in data_dict['image_data'].items():
          names.append(name)
          images.append(image)
        
        # preprocess
        # image name should be specified
        if self.cfg.IMAGE.NAME in data_dict['image_data'].keys():
            image = cvt_image(data_dict['image_data'][self.cfg.IMAGE.NAME], 
                              data_dict['infos']['image_param'][self.cfg.IMAGE.NAME]['w'] // 32 * 32, 
                              data_dict['infos']['image_param'][self.cfg.IMAGE.NAME]['h'] // 32 * 32)
        else:
            self.cfg.IMAGE.NAME, image = data_dict['image_data'].popitem()
            image = cvt_image(image, 
                              data_dict['infos']['image_param'][self.cfg.IMAGE.NAME]['w'] // 32 * 32, 
                              data_dict['infos']['image_param'][self.cfg.IMAGE.NAME]['h'] // 32 * 32)
        
        image_infer = np.expand_dims(image, axis=0).astype(np.float32)

        # cnn inference
        pred = self.engine.run([image_infer])

        # postprocess
        pred_dicts = self.post_processer.forward(pred)
        pred_dicts = self.post_processer.detect_colors(image, pred_dicts)
        pred_dicts['image_name'] = self.cfg.IMAGE.NAME

        trafficlight_result = {"trafficlight": pred_dicts}
        data_dict['infos'].update(trafficlight_result)
        result = {'trafficlight' : data_dict['infos']}
        return result