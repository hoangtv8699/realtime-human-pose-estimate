import cv2
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import yaml
from easydict import EasyDict

from utils.pose_resnet_util import _boxs2cs, get_final_preds, get_affine_transform, _box2cs
from tool.darknet2pytorch import Darknet
from tool.torch_utils import *
from tool.utils import *
from utils.pose_resnet_model import *
import time

# yolo cfg
yolov4_cfg = 'models/yolov4/yolov4-tiny.cfg'
yolov4_wt = 'models/yolov4/yolov4-tiny.weights'
yolov4_labels = 'models/yolov4/coco.names'

# load pre-trained yolo model
yolov4 = Darknet(yolov4_cfg)
yolov4.load_weights(yolov4_wt)
yolov4.cuda()
yolov4.eval()

class_names = load_class_names(yolov4_labels)

# PoseResnet cfg
resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}

pose_resnet_cfg = open('models/pose/384x288_d256x3_adam_lr1e-3.yaml')
pose_resnet_cfg = yaml.load(pose_resnet_cfg, Loader=yaml.FullLoader)
pose_resnet_cfg = EasyDict(pose_resnet_cfg)

num_layers = pose_resnet_cfg.MODEL.EXTRA.NUM_LAYERS
block_class, layers = resnet_spec[num_layers]
pose_resnet = PoseResNet(block_class, layers, pose_resnet_cfg)

# load pre-trained PoseResnet
checkpoint = torch.load('models/pose/pose_resnet_50_384x288.pth.tar', map_location=torch.device('cpu'))
pose_resnet.load_state_dict(checkpoint)
pose_resnet.cuda()
pose_resnet.eval()

# load video
cap = cv2.VideoCapture('videos/ellentube.mp4')
cap.set(3, 1280)
cap.set(4, 720)
while (True):
    t1 = time.time()
    ret, ori_img = cap.read()
    # load img
    img = cv2.resize(ori_img, (yolov4.width, yolov4.height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # detect box and format box to x, y, w, h
    boxes = do_detect(yolov4, img, 0.35, 0.6, True)

    width = ori_img.shape[1]
    height = ori_img.shape[0]
    # print(pose_resnet_cfg.MODEL.IMAGE_SIZE)
    human_boxes = []
    for box in boxes[0]:
        if box[6] == 0:
            x1 = int(box[0] * width)
            y1 = int(box[1] * height)
            x2 = int(box[2] * width)
            y2 = int(box[3] * height)

            if x1 < 0:
                x1 = 0

            if x2 > width:
                x2 = width

            if y1 < 0:
                y1 = 0

            if y2 > height:
                y2 = height

            human_boxes.append([x1, y1, x2, y2])

    cs = []
    ss = []
    inputs = []
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    for box in human_boxes:
        x1, y1, x2, y2 = box
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

    # with torch.no_grad():
    # compute output heatmap
    output = pose_resnet(inputs.cuda())
    # compute coordinate
    t3 = time.time()
    preds, maxvals = get_final_preds(
        output.detach().cpu().numpy(), np.asarray(cs), np.asarray(ss)
    )
    t4 = time.time()
    # plot
    for pred, box in zip(preds, human_boxes):
        shift_x = box[0]
        shift_y = box[1]
        for mat in pred:
            x, y = int(mat[0]), int(mat[1])
            cv2.circle(ori_img, (x + shift_x, y + shift_y), 2, (255, 0, 0), 2)
        # cv2.rectangle(ori_img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 1)

    # vis result
    t2 = time.time()
    print('time process: ', t2 - t1)
    # print('time pose process: ', t4 - t3)
    cv2.imshow('frame', ori_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
