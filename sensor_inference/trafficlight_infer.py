import copy
from sensor_inference.infer_base import InferBase
from sensor_inference.utils.config import cfg, cfg_from_yaml_file

from util.image_util import cvt_image

def parse_config(cfg_file):
    cfg_from_yaml_file(cfg_file, cfg)
    return cfg

def build_engine(onnx_file, trt_file):
    from sensor_inference.trt_engine.runtime_engine import RuntimeBackend as backend
    engine = backend(trt_path=trt_file)
    return engine

class TrafficLightInfer(InferBase):
    def __init__(self, engine_start, cfg_file = None, logger = None, max_size = 3):
        super().__init__('TrafficLight', engine_start, cfg_file, logger, max_size)

    def initialize(self):
        self.cfg = copy.deepcopy(parse_config(self.cfg_file))
        self.create_queue()

    def build_engine(self, calib):
        from sensor_inference.utils.trafficlight_post_process import PostProcesser
        self.engine = build_engine(self.cfg.ONNX_FILE, self.cfg.TRT_FILE)
        self.post_processer = PostProcesser()

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

        image = data_dict['image_data']

        # cnn
        pred = self.engine.run([image])

        # postprocess
        pred_dicts = self.post_processer.forward(pred)

        data_dict['infos'].update(pred_dicts)
        result = {'trafficlight' : data_dict['infos']}
        return result