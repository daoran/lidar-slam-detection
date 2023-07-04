import sys
import os
sys.path.append(os.getcwd())

import argparse

import numpy as np
import torch
import torch.onnx
import cv2
import onnx
from onnxsim import simplify

from sensor_inference.utils.config import cfg, cfg_from_yaml_file
from sensor_inference.pytorch_model.image_model.image_detection import ImageDetection

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='sensor_inference/cfgs/detection_image.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')

    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg

def main():
    args, cfg = parse_config()

    model = ImageDetection(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES))
    model.load_params_from_file(filename=args.ckpt, to_cpu=True)
    model.cuda()
    model.eval()

    param = dict()
    param['name'] = '0'
    param['height'] = 352
    param['width'] = 640
    param['extrinsic_parameters'] = [0, 0, 0, -90, 0, 90]
    param['intrinsic_parameters'] = [7.215377e+02, 7.215377e+02, 6.095593e+02, 1.72854e+02, 0 , 0, 0, 0]

    with torch.no_grad():
        image = cv2.imread(args.data_path)
        images = np.expand_dims(image, axis=0)
        images = torch.from_numpy(images)
        images = images.float().cuda()

        hm, kps, dim, rot, score_preds, label_preds, indices = model(images)
        hm  = hm.cpu().numpy()
        kps = kps.cpu().numpy()
        dim = dim.cpu().numpy()
        rot = rot.cpu().numpy()
        score_preds = score_preds.cpu().numpy()
        label_preds = label_preds.cpu().numpy()
        indices = indices.cpu().numpy()

        input_names = ["image"]
        output_names = ["hm", "kps", "dim", "rot", "score_preds", "label_preds", "indices"]
        torch.onnx.export(model, images, cfg.ONNX_FILE, verbose=False, export_params=True, opset_version=10, input_names=input_names, output_names=output_names)

        onnx_model = onnx.load(cfg.ONNX_FILE)
        model_simp, check = simplify(onnx_model, dynamic_input_shape=False, input_shapes=None)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, cfg.ONNX_FILE)
        print("export done")

if __name__ == '__main__':
    main()