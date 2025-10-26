import csv
import os
import argparse
from ultralytics import YOLO
import numpy as np
import cv2
from tqdm import tqdm
from scripts.merge_nms import nms_merge
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--weights', default='../models/experiments/yolov8_crop_detect/weights/best.pt')
parser.add_argument('--imgdir', default='../data/images/test')
parser.add_argument('--gt_csv', default='../data/test_counts.csv')
parser.add_argument('--conf', type=float, default=0.25)
args = parser.parse_args()

model = YOLO(args.weights)

# load GT
gt = {}

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


# Load ground truth data
with open(args.gt_csv, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) >= 2:
            gt[row[0]] = int(row[1])

preds = []
trues = []

for img_name, true_count in gt.items():
    img_path = os.path.join(args.imgdir, img_name)
    if not os.path.exists(img_path):
        print(f"Warning: Image {img_name} not found, skipping...")
        continue
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not load image {img_name}, skipping...")
        continue
    
    tiles, coords = tile_image(img, tile_size=1024, overlap=200)
    
    all_boxes = []
    all_scores = []
    all_classes = []
    
    for tile, (x_off, y_off) in zip(tiles, coords):
        res = model.predict(source=tile, conf=args.conf, task='detect', verbose=False)
        r = res[0]
        if len(r.boxes) == 0:
            continue
        
        boxes = r.boxes.xyxy.numpy()
        scores = r.boxes.conf.numpy()
        classes = r.boxes.cls.numpy()
        
        for b, s, c in zip(boxes, scores, classes):
            x1, y1, x2, y2 = b
            all_boxes.append([float(x1 + x_off), float(y1 + y_off), float(x2 + x_off), float(y2 + y_off)])
            all_scores.append(float(s))
            all_classes.append(int(c))
    
    # Apply NMS to merge overlapping detections
    final_boxes, final_scores, final_classes = nms_merge(all_boxes, all_scores, all_classes, iou_thresh=0.5)
    
    pred_count = len(final_boxes)
    preds.append(pred_count)
    trues.append(true_count)
    
    print(f"{img_name}: Predicted={pred_count}, True={true_count}")

# Calculate metrics
mae = np.mean(np.abs(np.array(preds) - np.array(trues)))
rmse = np.sqrt(np.mean((np.array(preds) - np.array(trues))**2))
r2 = 1 - (np.sum((np.array(trues) - np.array(preds))**2) / np.sum((np.array(trues) - np.mean(np.array(trues)))**2))

print(f"\nEvaluation Results:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.3f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(trues, preds, alpha=0.7)
plt.plot([min(trues), max(trues)], [min(trues), max(trues)], 'r--', label='Perfect prediction')
plt.xlabel('True Count')
plt.ylabel('Predicted Count')
plt.title('True vs Predicted Counts')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('evaluation_plot.png')
plt.show()


