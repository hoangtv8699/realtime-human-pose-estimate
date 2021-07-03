import torch
from utils.pose_resnet import *
import yaml
from easydict import EasyDict
import cv2
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import json

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
# model.cuda()
model.eval()

# img = cv2.imread('images/100031.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
# img = cv2.resize(img, (384, 288))
#
# img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
# img.cuda()
#
# out = model(img)
# print(out.size())
#
# h, w = img.shape[0], img.shape[1]
#
# print(img.shape)
#
# tl = [10, 400]
# br = [300, 100]

json_data = open('F:/Ths/CV/crowdpose_dataset/anotation/crowdpose_val.json', 'r')
json_data = json.load(json_data)

print(json_data.keys())
images_list = json_data['images']
annotations_list = json_data['annotations']
categories_list = json_data['categories']

print('images lenth:', len(images_list))
print('annotations lenth:', len(annotations_list))
print('categories lenth:', len(categories_list))

print('categories:', categories_list)
print(images_list[0].keys())
print(annotations_list[0].keys())


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


def vis_box(img, bboxes):
    for box in bboxes:
        x0, y0, w, h = box
        img = cv2.rectangle(img, (x0, y0 + h), (x0 + w, y0), color=[0, 255, 0], thickness=2, lineType=cv2.LINE_AA)  # 12
    return img


def vis_keypoints(img, kpts, crowdIndex):
    links = [[0, 2], [1, 3], [2, 4], [3, 5], [6, 8], [8, 10], [7, 9], [9, 11], [12, 13], [0, 13], [1, 13], [6, 13],
             [7, 13]]
    for kpt in kpts:
        x_ = kpt[0::3]
        y_ = kpt[1::3]
        v_ = kpt[2::3]
        for order1, order2 in links:
            if v_[order1] > 0 and v_[order2] > 0:
                img = img = cv2.line(img, (x_[order1], y_[order1]), (x_[order2], y_[order2]), color=[100, 255, 255],
                                     thickness=2, lineType=cv2.LINE_AA)
        for x, y, v in zip(x_, y_, v_):
            if int(v) > 0:
                img = cv2.circle(
                    img, (int(x), int(y)),
                    radius=3, color=[255, 0, 255], thickness=-1, lineType=cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.putText(img, 'crowdIndex: ' + str(crowdIndex), (0, 50), font, 1.2, (0, 0, 0), 2)

    return img


vis_id, file_name, crowdIndex = images_list[0]['id'], images_list[0]['file_name'], images_list[0]['crowdIndex']
print(file_name)
print(crowdIndex)

kpts, bboxes = get_annokpts(vis_id, annotations_list)
img = cv2.imread('F:/Ths/CV/crowdpose_dataset/images/' + file_name)
pose_img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
# print(kpts)
print(bboxes)

x0, y0, w, h = bboxes[0]

top_left = [x0, y0 + h]
bottom_right = [x0 + w, y0]



