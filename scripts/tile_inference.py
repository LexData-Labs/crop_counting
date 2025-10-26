import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from scripts.merge_nms import nms_merge
from tqdm import tqdm




def tile_image(img, tile_size=1024, overlap=200):
    h, w = img.shape[:2]
    stride = tile_size - overlap
    tiles = []
    coords = []
    ys = list(range(0, max(1, h - tile_size + 1), stride))
    xs = list(range(0, max(1, w - tile_size + 1), stride))
    if (h - tile_size) % stride != 0:
        ys.append(h - tile_size)
    if (w - tile_size) % stride != 0:
        xs.append(w - tile_size)
    for y in ys:
        for x in xs:
            tile = img[y:y+tile_size, x:x+tile_size]
            tiles.append(tile)
            coords.append((x, y))
    return tiles, coords




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True)
    parser.add_argument('--weights', default='../models/experiments/yolov8_crop_detect/weights/best.pt')
    parser.add_argument('--tile', type=int, default=1024)
    parser.add_argument('--overlap', type=int, default=200)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.45)
    parser.add_argument('--merge_iou', type=float, default=0.5)
    args = parser.parse_args()


model = YOLO(args.weights)
img = cv2.imread(args.img)
tiles, coords = tile_image(img, tile_size=args.tile, overlap=args.overlap)


all_boxes = []
all_scores = []
all_classes = []


for tile, (x_off, y_off) in tqdm(list(zip(tiles, coords)), total=len(tiles)):
    res = model.predict(source=tile, conf=args.conf, iou=args.iou, task='detect', verbose=False)
    r = res[0]
    if len(r.boxes) == 0:
        continue
    boxes = r.boxes.xyxy.numpy()  # shape (n,4)
    scores = r.boxes.conf.numpy()
    classes = r.boxes.cls.numpy()
    for b, s, c in zip(boxes, scores, classes):
        x1, y1, x2, y2 = b
        all_boxes.append([float(x1 + x_off), float(y1 + y_off), float(x2 + x_off), float(y2 + y_off)])
        all_scores.append(float(s))
        all_classes.append(int(c))


final_boxes, final_scores, final_classes = nms_merge(all_boxes, all_scores, all_classes, iou_thresh=args.merge_iou)
print('Final count:', len(final_boxes))


# optionally draw final boxes
for b in final_boxes:
    x1, y1, x2, y2 = map(int, b)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imwrite('tile_inference_result.png', img)
print('Saved annotated orthomosaic to tile_inference_result.png')