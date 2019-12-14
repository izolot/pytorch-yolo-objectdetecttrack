import sys
import cv2
import time
import numpy as np
import torch

from PIL import Image
from torchvision import transforms
from models import *
from sort import *
from utils import *

# load weights and set defaults
config_path = 'pytorch-yolo-objectdetecttrack/config/yolov3.cfg'
weights_path = 'pytorch-yolo-objectdetecttrack/yolov3.weights'
class_path = 'pytorch-yolo-objectdetecttrack/config/coco.names'
img_size = 416
conf_thres = 0.6
nms_thres = 0.4

bags_classes = [24, 25, 26, 28]
DETECT_STATIC = True
occup_rate = 0.1

# load model and put into eval mode
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.cuda()
model.eval()

back_sub = cv2.createBackgroundSubtractorMOG2()

classes = utils.load_classes(class_path)
Tensor = torch.cuda.FloatTensor


def detect_image(img):
    # scale and pad image
    ratio = min(img_size / img.size[0], img_size / img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([transforms.Resize((imh, imw)),
                                         transforms.Pad((max(int((imh - imw) / 2), 0), max(int((imw - imh) / 2), 0),
                                                         max(int((imh - imw) / 2), 0), max(int((imw - imh) / 2), 0)),
                                                        (128, 128, 128)),
                                         transforms.ToTensor(),
                                         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(
            detections, 80, conf_thres, nms_thres)
    return detections[0]


videopath = ""

color = 0, 200, 0

cap = cv2.VideoCapture(videopath)
mot_tracker = Sort()

cv2.namedWindow('Stream', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Stream', (800, 600))

frames = 0
starttime = time.time()
alarm_seconds = 20
static_obj = {}
abandoned_obj = {}
while cap.isOpened():
    time_iter = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    frames += 1
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    obj_mask_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fg_mask = back_sub.apply(frame)
    pilimg = Image.fromarray(frame)
    detections = detect_image(pilimg)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    img = np.array(pilimg)
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x
    if detections is not None:
        tracked_objects = mot_tracker.update(detections.cpu())

        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
            if int(cls_pred) in bags_classes:
                box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
                x2 = x1 + box_w
                y2 = y1 + box_h
                cls = classes[int(cls_pred)]
                obj_id = int(obj_id)
                label = cls + "-" + str(int(obj_id))
                if DETECT_STATIC:
                    fg_mask_obj = fg_mask[y1:(y2 + 1), x1:(x2 + 1)]
                    size_mask = np.size(fg_mask_obj)
                    val_of_tresh = np.count_nonzero(fg_mask_obj)
                    print("x1 {} y1 {} x2 {} y2 {}".format(x1, y1, x2, y2))
                    print("id {} class {} size of mask {} nonzero {}"
                          .format(int(obj_id), cls, size_mask, val_of_tresh))
                    if val_of_tresh < size_mask * occup_rate:
                        color = 0, 0, 200
                        if obj_id in static_obj.keys():
                            static_obj[obj_id]["x_y"] = (x1, y1, x2, y2)
                            static_obj[obj_id]["count"] += 1
                            # if object has static condition already 17 iter ~ 1 sec:
                            if static_obj[obj_id]["count"] > 17 * alarm_seconds:
                                label = "ATTENTION"
                        else:
                            static_obj[obj_id] = {"x_y": (x1, y1, x2, y2), "count": 0}
                    else:
                        color = 0, 200, 0

                    obj_mask_frame[y1:(y2 + 1), x1:(x2 + 1)] = fg_mask_obj

                cv2.rectangle(frame, (x1, y1),
                            (x2, y2), color, 4)
                cv2.rectangle(frame, (x1, y1 - 35),
                            (x1 + len(cls) * 19 + 80, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 3)
    print("FPS_TIME {}".format(1/(time.time() - time_iter)))
    cv2.imshow('Stream', frame)
    #cv2.imshow('FG Mask', fg_mask)
    cv2.imshow('Object Mask', obj_mask_frame)
    #outvideo.write(frame)
    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
        break

totaltime = time.time() - starttime
print(frames, "frames", totaltime / frames, "s/frame")
cv2.destroyAllWindows()
#outvideo.release()
