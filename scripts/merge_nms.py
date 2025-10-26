import numpy as np


def nms_merge(boxes, scores, classes, iou_thresh=0.5):
    # boxes: Nx4 array [[x1,y1,x2,y2]]
    if len(boxes) == 0:
        return [], [], []
    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)
    keep_boxes = []
    keep_scores = []
    keep_classes = []
    for c in np.unique(classes):
        idxs = np.where(classes == c)[0]
        b = boxes[idxs]
        s = scores[idxs]
        order = s.argsort()[::-1]
        while order.size > 0:
            i = order[0]
            keep_boxes.append(b[i].tolist())
            keep_scores.append(s[i].item())
            keep_classes.append(int(c))
            # compute IoU
            xx1 = np.maximum(b[i,0], b[:,0])
            yy1 = np.maximum(b[i,1], b[:,1])
            xx2 = np.minimum(b[i,2], b[:,2])
            yy2 = np.minimum(b[i,3], b[:,3])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            area_i = (b[i,2]-b[i,0])*(b[i,3]-b[i,1])
            area_all = (b[:,2]-b[:,0])*(b[:,3]-b[:,1])
            union = area_i + area_all - inter
            iou = inter / (union + 1e-6)
            inds = np.where(iou <= iou_thresh)[0]
            order = order[inds]
    return keep_boxes, keep_scores, keep_classes