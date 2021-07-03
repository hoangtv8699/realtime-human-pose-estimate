from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import cv2

model_cfg = 'models/yolov4/yolov4.cfg'
model_wt = 'models/yolov4/yolov4.weights'
model_labels = 'models/yolov4/coco.names'
imgfile = 'images/100031.jpg'

yolov4 = Darknet(model_cfg)
yolov4.load_weights(model_wt)

yolov4.cuda()

class_names = load_class_names(model_labels)

img = cv2.imread(imgfile)
sized = cv2.resize(img, (yolov4.width, yolov4.height))
sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

for i in range(2):
    start = time.time()
    boxes = do_detect(yolov4, sized, 0.4, 0.6, True)
    finish = time.time()
    if i == 1:
        print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

plot_boxes_cv2(img, boxes[0], savename='predictions.jpg', class_names=class_names)