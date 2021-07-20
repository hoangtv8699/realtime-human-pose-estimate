import math

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import torchvision.transforms as transforms


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_final_preds(batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # Transform back
    for i in range(coords.shape[0]):
        coords[i] = transform_preds(coords[i], center[i], scale[i],
                                    [heatmap_width, heatmap_height])
    return coords, maxvals


def _boxs2cs(boxs, image_width, image_height):
    cs = []
    ss = []
    for box in boxs:
        c, s = _box2cs(box, image_width, image_height)
        cs.append(c)
        ss.append(s)
    return cs, ss


def _box2cs(box, image_width, image_height):
    x, y, w, h = box[:4]
    return _xywh2cs(x, y, w, h, image_width, image_height)


def _xywh2cs(x, y, w, h, image_width, image_height):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    aspect_ratio = image_width * 1.0 / image_height
    pixel_std = 200

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
        dtype=np.float32)

    return center, scale


def yolo_preprocessing(ori_img, human_boxes, pose_resnet_cfg):
    cs = []
    ss = []
    inputs = []

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    for box in human_boxes:
        x1, y1, x2, y2 = box[:4]
        c, s = _box2cs([0, 0, x2 - x1, y2 - y1], 288, 384)
        trans = get_affine_transform(c, s, 0, pose_resnet_cfg.MODEL.IMAGE_SIZE)
        cs.append(c)
        ss.append(s)
        input = cv2.warpAffine(
            ori_img[y1:y2, x1:x2],
            trans,
            pose_resnet_cfg.MODEL.IMAGE_SIZE,
            flags=cv2.INTER_LINEAR
        )
        inputs.append(transform(input))
    inputs = torch.stack(inputs).float()
    return cs, ss, inputs


def draw_all(ori_img, preds, maxvals, human_boxes):
    threshold = 0.3
    stickwidth = 4
    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [1, 18], [18, 12], [18, 13], [6, 18], [7, 18],
                [6, 8], [7, 9], [8, 10], [9, 11], [1, 2], [1, 3], [2, 4], [3, 5]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 170], [255, 0, 255], [255, 0, 85]]

    for pred, maxval, box in zip(preds, maxvals, human_boxes):
        pred = np.append(pred, [(pred[5] + pred[6]) / 2], axis=0)
        maxval = np.append(maxval, [(maxval[5] + maxval[6]) / 2], axis=0)
        shift_x = box[0]
        shift_y = box[1]
        for i in range(len(pred)):
            if maxval[i] < threshold:
                continue
            x, y = int(pred[i][0]), int(pred[i][1])
            cv2.circle(ori_img, (x + shift_x, y + shift_y), 4, colors[i], thickness=-1)
        if len(box) > 4:
            cv2.putText(ori_img, "id: {}".format(box[4]), (shift_x, shift_y), cv2.FONT_HERSHEY_PLAIN, 2,
                        [255, 255, 255], 2)
        # cv2.rectangle(ori_img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 1)
        for pair in skeleton:
            xy0 = pred[pair[0] - 1]
            xy1 = pred[pair[1] - 1]

            if maxval[pair[0] - 1] < threshold or maxval[pair[1] - 1] < threshold:
                continue

            xy0 = [xy0[0] + shift_x, xy0[1] + shift_y]
            xy1 = [xy1[0] + shift_x, xy1[1] + shift_y]

            mX = (xy0[0] + xy1[0]) / 2
            mY = (xy0[1] + xy1[1]) / 2

            length = ((xy0[0] - xy1[0]) ** 2 + (xy0[1] - xy1[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(xy0[1] - xy1[1], xy0[0] - xy1[0]))

            polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(ori_img, polygon, colors[pair[0] - 1])

    return ori_img
