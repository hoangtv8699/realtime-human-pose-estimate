import cv2
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import yaml
from easydict import EasyDict

from pose_resnet.pose_resnet_util import _boxs2cs, get_final_preds, get_affine_transform
from yolo.darknet2pytorch import Darknet
from yolo.torch_utils import *
from yolo.utils import *
from pose_resnet.pose_resnet_model import *

# yolo cfg
yolov4_cfg = 'models/yolov4/yolov4.cfg'
yolov4_wt = 'models/yolov4/yolov4.weights'
yolov4_labels = 'models/yolov4/coco.names'

# load pre-trained yolo model
yolov4 = Darknet(yolov4_cfg)
yolov4.load_weights(yolov4_wt)
# yolov4.cuda()
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
# pose_resnet.cuda()
pose_resnet.eval()

# load img
imgfile = 'images/100038.jpg'
ori_img = cv2.imread(imgfile)
img = cv2.resize(ori_img, (yolov4.width, yolov4.height))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

if type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
    img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
elif type(img) == np.ndarray and len(img.shape) == 4:
    img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().div(255.0)
else:
    print("unknow image type")
    exit(-1)

# detect box and format box to x, y, w, h
boxes = do_detect(yolov4, img, 0.4, 0.6, False)

width = ori_img.shape[1]
height = ori_img.shape[0]
print(pose_resnet_cfg.MODEL.IMAGE_SIZE)
human_boxes = []
for box in boxes[0]:
    if box[6] == 0:
        x1 = int(box[0] * width)
        y1 = int(box[1] * height)
        x2 = int(box[2] * width)
        y2 = int(box[3] * height)
        human_boxes.append([x1, y1, x2 - x1, y2 - y1])
print(human_boxes)

tmp = ori_img.copy()
for box in human_boxes:
    x1, y1, x2, y2 = box
    tmp = cv2.rectangle(tmp, (x1, y1), (x1 + x2, y1 + y2), (255, 0, 0), 1)
    # tmp = cv2.circle(tmp, (x1, y1), 2, (255, 0, 0), 2)
    # break
plt.imshow(cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB))
plt.show()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

cs, ss = _boxs2cs(human_boxes, 288, 384)
inputs = []
for c, s in zip(cs, ss):
    trans = get_affine_transform(c, s, 0, pose_resnet_cfg.MODEL.IMAGE_SIZE)
    input = cv2.warpAffine(
        ori_img,
        trans,
        pose_resnet_cfg.MODEL.IMAGE_SIZE,
        flags=cv2.INTER_LINEAR
    )
    plt.imshow(cv2.cvtColor(input, cv2.COLOR_BGR2RGB))
    plt.show()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    input = (input / 255.0 - mean) / std
    inputs.append(input.transpose(2, 0, 1))


inputs = torch.from_numpy(np.asarray(inputs)).float()
# transform = transforms.Compose([
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225]),
# ])
# inputs = transform(inputs)

with torch.no_grad():
    # compute output heatmap
    t3 = time.time()
    output = pose_resnet(inputs)
    t4 = time.time()
    # compute coordinate
    preds, maxvals = get_final_preds(
        output.detach().cpu().numpy(), np.asarray(cs), np.asarray(ss))
    # plot
    image = ori_img.copy()
    for pred in preds:
        for mat in pred:
            x, y = int(mat[0]), int(mat[1])
            cv2.circle(image, (x, y), 2, (255, 0, 0), 2)

    # vis result
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

