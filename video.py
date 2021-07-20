import time
import cv2
import matplotlib.pyplot as plt

import yaml
from easydict import EasyDict

from pose_resnet.pose_resnet_util import *
from yolo.darknet2pytorch import Darknet
from yolo.torch_utils import *
from yolo.utils import *
from pose_resnet.pose_resnet_model import *

from deep_sort.deep_sort import DeepSort

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

# initialize deep sort objec
deepsort = DeepSort('models/deepSORT/original_ckpt.t7')

# load video
cap = cv2.VideoCapture('videos/ellentube_Trim.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)

result = cv2.VideoWriter('ellentube_Trim_output.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)

# out = cv2.VideoWriter('output.avi', -1, 20.0, (640,480))

time_consum = []

while (True):
    t1 = time.time()
    ret, ori_img = cap.read()
    if not ret:
        break
    # load img
    img = cv2.resize(ori_img, (yolov4.width, yolov4.height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # detect box and format box to x, y, w, h
    boxes = do_detect(yolov4, img, 0.3, 0.5, True)
    human_boxes, conf = to_origin_box(ori_img, boxes)

    if len(human_boxes) < 1:
        cv2.imshow('frame', ori_img)
        continue
    track_outputs = deepsort.update(human_boxes, conf, ori_img)

    if len(track_outputs) > 0:
        human_boxes = track_outputs

    cs, ss, inputs = yolo_preprocessing(ori_img, human_boxes, pose_resnet_cfg)

    output = pose_resnet(inputs.cuda())
    # compute coordinate
    preds, maxvals = get_final_preds(
        output.cpu().detach().numpy(), np.asarray(cs), np.asarray(ss)
    )

    ori_img = draw_all(ori_img, preds, maxvals, human_boxes)
    t2 = time.time()
    # average fps 10 frameD
    time_consum.append(t2 - t1)
    if len(time_consum) > 10:
        time_consum.pop(0)
    # print(np.mean(time_consum))
    fps = int(10 / (np.mean(time_consum))) / 10
    cv2.putText(ori_img, "fps: {}".format(str(fps)), (7, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, [255, 255, 255], 2)
    result.write(ori_img)
    cv2.imshow('frame', ori_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
result.release()
cv2.destroyAllWindows()