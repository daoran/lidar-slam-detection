import numpy as np
import cv2
from sensor_inference.utils import model_nms_utils
from util.box_utils import boxes3d_camera_to_lidar

def gen_3box_np(kps, dim, rot, calib, opinv, const):
    c=kps.shape[0]

    opinv = np.tile(opinv, (c, 1, 1))
    kps = np.transpose(kps.reshape(c, -1, 2), (0, 2, 1))
    hom = np.ones((c, 1, 9))
    kps = np.concatenate((kps, hom), axis=1)
    kps = np.matmul(opinv, kps).reshape(c, 2, 9)
    kps = np.transpose(kps, (0, 2, 1)).reshape(c, -1)
    si = np.zeros_like(kps[:, 0:1]) + calib[0:1, 0:1]
    rots = rot[..., 0:1]
    rotc = rot[..., 1:2]
    alpha = np.arctan2(rots, rotc)

    rot_y = alpha + np.arctan2(kps[:, 16:17] - calib[0:1, 2:3], si)
    rot_y[rot_y > np.pi] = rot_y[rot_y > np.pi] - 2 * np.pi
    rot_y[rot_y < - np.pi] = rot_y[rot_y < - np.pi] + 2 * np.pi

    calib = np.tile(calib, (c, 1, 1))
    kpoint = kps[:, :16]
    f = np.expand_dims(calib[:, 0, 0], axis=1)
    f = np.tile(f, (1, 16))
    cx, cy = np.expand_dims(calib[:, 0, 2], axis=1), np.expand_dims(calib[:, 1, 2], axis=1)
    cxy = np.concatenate((cx, cy), axis=1)
    cxy = np.tile(cxy, (1, 8))
    kp_norm = (kpoint - cxy) / f

    l = dim[:, 2:3]
    h = dim[:, 0:1]
    w = dim[:, 1:2]
    cosori = np.cos(rot_y)
    sinori = np.sin(rot_y)

    B = np.zeros_like(kpoint)
    C = np.zeros_like(kpoint)

    kp = np.expand_dims(kp_norm, axis=2)
    const = np.tile(const, (c, 1, 1))
    A = np.concatenate([const, kp], axis=2)

    B[:, 0:1] = l * 0.5 * cosori + w * 0.5 * sinori
    B[:, 1:2] = h * 0.5
    B[:, 2:3] = l * 0.5 * cosori - w * 0.5 * sinori
    B[:, 3:4] = h * 0.5
    B[:, 4:5] = -l * 0.5 * cosori - w * 0.5 * sinori
    B[:, 5:6] = h * 0.5
    B[:, 6:7] = -l * 0.5 * cosori + w * 0.5 * sinori
    B[:, 7:8] = h * 0.5
    B[:, 8:9] = l * 0.5 * cosori + w * 0.5 * sinori
    B[:, 9:10] = -h * 0.5
    B[:, 10:11] = l * 0.5 * cosori - w * 0.5 * sinori
    B[:, 11:12] = -h * 0.5
    B[:, 12:13] = -l * 0.5 * cosori - w * 0.5 * sinori
    B[:, 13:14] = -h * 0.5
    B[:, 14:15] = -l * 0.5 * cosori + w * 0.5 * sinori
    B[:, 15:16] = -h * 0.5

    C[:, 0:1] = -l * 0.5 * sinori + w * 0.5 * cosori
    C[:, 1:2] = -l * 0.5 * sinori + w * 0.5 * cosori
    C[:, 2:3] = -l * 0.5 * sinori - w * 0.5 * cosori
    C[:, 3:4] = -l * 0.5 * sinori - w * 0.5 * cosori
    C[:, 4:5] = l * 0.5 * sinori - w * 0.5 * cosori
    C[:, 5:6] = l * 0.5 * sinori - w * 0.5 * cosori
    C[:, 6:7] = l * 0.5 * sinori + w * 0.5 * cosori
    C[:, 7:8] = l * 0.5 * sinori + w * 0.5 * cosori
    C[:, 8:9] = -l * 0.5 * sinori + w * 0.5 * cosori
    C[:, 9:10] = -l * 0.5 * sinori + w * 0.5 * cosori
    C[:, 10:11] = -l * 0.5 * sinori - w * 0.5 * cosori
    C[:, 11:12] = -l * 0.5 * sinori - w * 0.5 * cosori
    C[:, 12:13] = l * 0.5 * sinori - w * 0.5 * cosori
    C[:, 13:14] = l * 0.5 * sinori - w * 0.5 * cosori
    C[:, 14:15] = l * 0.5 * sinori + w * 0.5 * cosori
    C[:, 15:16] = l * 0.5 * sinori + w * 0.5 * cosori

    B = B - kp_norm * C

    AT = np.transpose(A, (0, 2, 1))
    B = np.expand_dims(B, axis=2)

    pinv = np.matmul(AT, A)
    pinv = np.linalg.inv(pinv)

    pinv = np.matmul(pinv, AT)
    pinv = np.matmul(pinv, B)
    pinv = np.squeeze(pinv, axis=2)
    pred_boxes = np.concatenate([pinv, dim, rot_y], axis=1)

    return pred_boxes

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_affine_transform(center,
                         scale,
                         output_size,
                         inv=False):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    src_dir = get_dir([0, src_w * -0.5], 0)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def nms_np(dets, thresh):
    """
    nms
    :param dets: ndarray [x1,y1,x2,y2,score]
    :param thresh: int
    :return: list[index]
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    order = dets[:, 4].argsort()[::-1]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        over = (w * h) / (area[i] + area[order[1:]] - w * h)
        index = np.where(over <= thresh)[0]
        order = order[index + 1]
    return keep

class ImagePostProcesser():
    def __init__(self, model_cfg, num_class):
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.cam_param = dict()

        self.const = np.array(
                    [[-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1],
                    [-1, 0], [0, -1], [-1, 0], [0, -1]], dtype=np.float32)
        self.const = np.expand_dims(self.const, axis=0)

    def set_camera_parameter(self, cam_param):
        self.cam_param = cam_param.copy()
        for name, param in cam_param.items():
            w = param['w']
            h = param['h']

            c = np.array([w / 2., h / 2.], dtype=np.float32)
            s = max(h, w)
            output_w = w / self.model_cfg.OUTPUT_DOWN_SCALE
            output_h = h / self.model_cfg.OUTPUT_DOWN_SCALE
            trans_output = get_affine_transform(c, s, [output_w, output_h])
            trans_output_inv = get_affine_transform(c, s, [output_w, output_h], inv=True)

            self.cam_param[name]['trans_output'] = trans_output
            self.cam_param[name]['trans_output_inv'] = trans_output_inv
            self.cam_param[name]['output_w'] = output_w
            self.cam_param[name]['output_h'] = output_h

    def forward(self, name, hm, kps, dim, rot, score_preds, label_preds, indices):
        meta = self.cam_param[name]

        ys = (indices / meta['output_w']).astype(np.int32).astype(np.float32)
        xs = (indices % meta['output_w']).astype(np.int32).astype(np.float32)

        kps[:, ::2] = kps[:, ::2] + np.tile(np.expand_dims(xs, axis=1), (1, 9))
        kps[:, 1::2] = kps[:, 1::2] + np.tile(np.expand_dims(ys, axis=1), (1, 9))

        bboxes_kp = kps.reshape(-1, 9, 2)
        box_min = np.min(bboxes_kp, axis=1)
        box_max = np.max(bboxes_kp, axis=1)
        bboxes = np.concatenate((box_min, box_max, score_preds[..., np.newaxis]), axis=1)

        scores_mask = (score_preds >= self.model_cfg.SCORE_THRESH)
        kps = kps[scores_mask]
        dim = dim[scores_mask]
        rot = rot[scores_mask]
        pred_scores = score_preds[scores_mask]
        pred_labels = label_preds[scores_mask].astype(np.int32) + 1
        pred_bboxes = bboxes[scores_mask]

        keep = nms_np(pred_bboxes, 0.5)
        kps = kps[keep]
        dim = dim[keep]
        rot = rot[keep]
        pred_scores = pred_scores[keep]
        pred_labels = pred_labels[keep]
        pred_bboxes = pred_bboxes[keep]

        pred_boxes = self.post_processing(kps, dim, rot,
                                          meta['intrinsic'],
                                          meta['trans_output_inv'],
                                          self.const,
                                          meta['V2C'])

        select = model_nms_utils.class_agnostic_nms_camera(
            box_scores=pred_scores, box_preds=pred_boxes,
            nms_config=self.model_cfg.NMS_CONFIG,
        )
        pred_boxes = pred_boxes[select]
        pred_scores = pred_scores[select]
        pred_labels = pred_labels[select]
        pred_bboxes = pred_bboxes[select]

        for idx in range(pred_bboxes.shape[0]):
            pred_bboxes[idx, :2] = affine_transform(pred_bboxes[idx, :2], meta['trans_output_inv'])
            pred_bboxes[idx, 2:4] = affine_transform(pred_bboxes[idx, 2:4], meta['trans_output_inv'])

        pred_labels = pred_labels.reshape(pred_labels.shape[0], 1)
        pred_scores = pred_scores.reshape(pred_scores.shape[0], 1)
        pred_sensor = np.ones((pred_scores.shape[0], 1), dtype=np.int32)

        result_dict = {
            'pred_boxes': pred_boxes,
            'pred_scores': pred_scores,
            'pred_labels': pred_labels,
            'pred_sensor': pred_sensor,
            'pred_bboxes': pred_bboxes,
            'pred_heatmap': hm,
        }

        return result_dict

    def post_processing(self, kps, dim, rot, calib, opinv, const, V2C):
        if kps.shape[0] == 0:
            return np.empty([0, 7], dtype=np.float32)
        pred_boxes = gen_3box_np(kps, dim, rot, calib, opinv, const)
        pred_boxes[:, 1] += pred_boxes[:, 3] / 2.0
        pred_boxes = pred_boxes[:, [0, 1, 2, 5, 3, 4, 6]]
        pred_boxes = boxes3d_camera_to_lidar(pred_boxes, V2C)
        return pred_boxes