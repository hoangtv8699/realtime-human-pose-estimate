import sys
import time

import numpy as np
import cv2

from pose_resnet_util import *

from utils.pose_resnet import *
import yaml
from easydict import EasyDict
import json
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

COCO_CATEGORY = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]
THRESHOLD = 0.4
IOU = 0.45
POSE_THRESHOLD = 0.1


# ======================
# Display result
# ======================
def hsv_to_rgb(h, s, v):
    bgr = cv2.cvtColor(
        np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2]), 255


def pose_estimation(detector, pose, img):
    pose_img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    h, w = img.shape[0], img.shape[1]
    count = detector.get_object_count()
    pose_detections = []
    for idx in range(count):
        obj = detector.get_object(idx)
        top_left = (int(w * obj.x), int(h * obj.y))
        bottom_right = (int(w * (obj.x + obj.w)), int(h * (obj.y + obj.h)))
        CATEGORY_PERSON = 0
        if obj.category != CATEGORY_PERSON:
            pose_detections.append(None)
            continue
        px1, py1, px2, py2 = keep_aspect(
            top_left, bottom_right, pose_img, pose
        )
        crop_img = pose_img[py1:py2, px1:px2, :]
        offset_x = px1 / img.shape[1]
        offset_y = py1 / img.shape[0]
        scale_x = crop_img.shape[1] / img.shape[1]
        scale_y = crop_img.shape[0] / img.shape[0]
        detections = compute(
            pose, crop_img
        )
        pose_detections.append(detections)
    return pose_detections


def find_filename(img_id, meta):
    for block in meta:
        # print(block)
        if block['id'] == img_id:
            return block['file_name'], block['crowdIndex']
        continue
    return None, None


def get_annokpts(img_id, meta):
    kpts = []
    bboxes = []
    for block in meta:
        if block['image_id'] == img_id:
            kpts.append(block['keypoints'])
            bboxes.append(block['bbox'])
        continue
    return kpts, bboxes


# ======================
# Main functions
# ======================
def recognize_from_image():

    resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
                   34: (BasicBlock, [3, 4, 6, 3]),
                   50: (Bottleneck, [3, 4, 6, 3]),
                   101: (Bottleneck, [3, 4, 23, 3]),
                   152: (Bottleneck, [3, 8, 36, 3])}

    cfg = open('models/pose/384x288_d256x3_adam_lr1e-3.yaml')
    cfg = yaml.load(cfg, Loader=yaml.FullLoader)
    cfg = EasyDict(cfg)

    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS

    block_class, layers = resnet_spec[num_layers]

    pose = PoseResNet(block_class, layers, cfg)

    checkpoint = torch.load('models/pose/pose_resnet_50_384x288.pth.tar')
    pose.load_state_dict(checkpoint)
    # pose.cuda()
    pose.eval()

    json_data = open('F:/Ths/CV/crowdpose_dataset/anotation/crowdpose_val.json', 'r')
    json_data = json.load(json_data)

    images_list = json_data['images']
    annotations_list = json_data['annotations']

    vis_id, file_name, crowdIndex = images_list[0]['id'], images_list[0]['file_name'], images_list[0]['crowdIndex']

    kpts, bboxes = get_annokpts(vis_id, annotations_list)
    img = cv2.imread('F:/Ths/CV/crowdpose_dataset/images/' + file_name)
    pose_img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    x0, y0, w, h = bboxes[0]

    top_left = [x0, y0 + h]
    bottom_right = [x0 + w, y0]

    px1, py1, px2, py2 = keep_aspect(
        top_left, bottom_right, pose_img
    )
    px1, py1, px2, py2 = x0, y0, x0 + w, y0 + h

    crop_img = pose_img[py1:py2, px1:px2, :]
    # crop_img = pose_img

    detections = compute(
        pose, crop_img
    )
    return detections, crop_img, bboxes[0]


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
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


if __name__ == '__main__':
    # detections, img, box = recognize_from_image()
    # print(detections[0])
    #
    # for key in detections[0][0]:
    #     img = cv2.circle(img, (int(key[0]), int(key[1])), radius=3, color=[255, 0, 255], thickness=-1, lineType=cv2.LINE_AA)
    #
    # # x0, y0, w, h = box
    # # img = cv2.rectangle(img, (x0, y0 + h), (x0 + w, y0), color=[0, 255, 0], thickness=2, lineType=cv2.LINE_AA)
    #
    # plt.figure(figsize=(12, 10))
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.show()

    resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
                   34: (BasicBlock, [3, 4, 6, 3]),
                   50: (Bottleneck, [3, 4, 6, 3]),
                   101: (Bottleneck, [3, 4, 23, 3]),
                   152: (Bottleneck, [3, 8, 36, 3])}

    cfg = open('models/pose/384x288_d256x3_adam_lr1e-3.yaml')
    cfg = yaml.load(cfg, Loader=yaml.FullLoader)
    cfg = EasyDict(cfg)

    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS

    block_class, layers = resnet_spec[num_layers]

    model = PoseResNet(block_class, layers, cfg)

    checkpoint = torch.load('models/pose/pose_resnet_50_384x288.pth.tar')
    model.load_state_dict(checkpoint)
    # pose.cuda()
    model.eval()

    json_data = open('F:/Ths/CV/crowdpose_dataset/anotation/crowdpose_val.json', 'r')
    json_data = json.load(json_data)

    images_list = json_data['images']
    annotations_list = json_data['annotations']

    vis_id, file_name, crowdIndex = images_list[0]['id'], images_list[0]['file_name'], images_list[0]['crowdIndex']

    kpts, bboxes = get_annokpts(vis_id, annotations_list)
    img = cv2.imread('F:/Ths/CV/crowdpose_dataset/images/' + file_name)
    # pose_img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    x0, y0, w, h = bboxes[0]

    c, s = _box2cs(bboxes[0], img.shape[1], img.shape[0])
    r = 0

    trans = get_affine_transform(c, s, r, cfg.MODEL.IMAGE_SIZE)

    input = cv2.warpAffine(
        img,
        trans,
        (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
        flags=cv2.INTER_LINEAR)

    # cv2.imshow('image', input)
    # cv2.waitKey(3000)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    input = transform(input).unsqueeze(0)

    with torch.no_grad():
        # compute output heatmap
        output = model(input)
        # compute coordinate
        preds, maxvals = get_final_preds(
            output.clone().cpu().numpy(), np.asarray([c]), np.asarray([s]))
        # plot
        image = img.copy()
        for mat in preds[0]:
            x, y = int(mat[0]), int(mat[1])
            cv2.circle(image, (x, y), 2, (255, 0, 0), 2)

        # vis result
        cv2.imshow('res', image)
        cv2.waitKey(10000)





