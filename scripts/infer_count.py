import argparse
from ultralytics import YOLO
import cv2
import os


parser = argparse.ArgumentParser()
parser.add_argument('--weights', default='../models/experiments/yolov8_crop_detect/weights/best.pt')
parser.add_argument('--img', required=True)
parser.add_argument('--conf', type=float, default=0.25)
parser.add_argument('--iou', type=float, default=0.45)
parser.add_argument('--save', action='store_true')
args = parser.parse_args()


model = YOLO(args.weights)
res = model.predict(source=args.img, conf=args.conf, iou=args.iou, task='detect', verbose=False)
r = res[0]


# count is number of boxes after NMS
count = len(r.boxes)
print(f'Predicted count: {count}')


# save visualization
out = r.plot()  # ultralytics returns numpy image
if args.save:
    out_path = os.path.splitext(args.img)[0] + '_pred.png'
    cv2.imwrite(out_path, out)
    print('Saved annotated image to', out_path)