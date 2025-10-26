import os
import cv2
import numpy as np


def yolo_to_xyxy(yolo_box, img_w, img_h):
    # yolo_box: [x_center, y_center, w, h] normalized
    x_c, y_c, w, h = yolo_box
    x_c *= img_w; y_c *= img_h; w *= img_w; h *= img_h
    x1 = x_c - w/2; y1 = y_c - h/2; x2 = x_c + w/2; y2 = y_c + h/2
    return [x1, y1, x2, y2]




def xyxy_to_yolo(box, img_w, img_h):
    x1, y1, x2, y2 = box
    w = x2 - x1; h = y2 - y1
    x_c = x1 + w/2; y_c = y1 + h/2
    return [x_c/img_w, y_c/img_h, w/img_w, h/img_h]




def read_yolo_labels_for_image(label_path, img_shape):
    # returns list of xyxy boxes
    if not os.path.exists(label_path):
        return []
    h, w = img_shape[:2]
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5: 
                continue
            cls = int(parts[0])
            nums = list(map(float, parts[1:5]))
            xyxy = yolo_to_xyxy(nums, w, h)
            boxes.append((cls, xyxy))
    return boxes