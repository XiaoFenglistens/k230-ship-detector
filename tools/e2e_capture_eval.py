# SPDX-FileCopyrightText: 2026 XiaoFenglistens
# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""拍屏端到端逐框评测：把原图 GT 用单应性投影到照片坐标系，再与板端真实输出逐框算 IoU。

输入：
  <K230_ROOT>          干净照片（无绘制）
  <K230_ROOT>         板端同帧输出（x,y,w,h 为 800x480 坐标）
  datasets\\sodaa_ship_hbb\\images\\val\\00063__800__0___0.jpg   源图
  datasets\\sodaa_ship_hbb\\labels\\val\\00063__800__0___0.txt   GT
"""
import os as _os
KROOT = _os.environ.get("K230_ROOT", ".")  # 项目根目录，默认当前目录
import cv2, json, numpy as np, os

PHOTO = _os.path.join(KROOT, 'tmp', 'CAP_0005.jpg')
DETS = _os.path.join(KROOT, 'tmp', 'img_0005.json')
SRC_IMG = _os.path.join(KROOT, 'datasets', 'sodaa_ship_hbb', 'images', 'val', '00063__800__0___0.jpg')
SRC_LBL = _os.path.join(KROOT, 'datasets', 'sodaa_ship_hbb', 'labels', 'val', '00063__800__0___0.txt')
OUT = _os.path.join(KROOT, 'runs', 'capture_e2e')
os.makedirs(OUT, exist_ok=True)
IOU_T = 0.5

photo = cv2.imread(PHOTO)
src = cv2.imread(SRC_IMG)
H, W = src.shape[:2]
ph, pw = photo.shape[:2]

# ---------- 1. 单应性 ----------
sift = cv2.SIFT_create(nfeatures=6000)
k1, d1 = sift.detectAndCompute(cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY), None)
k2, d2 = sift.detectAndCompute(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), None)
bf = cv2.BFMatcher()
good = [a for a, b in bf.knnMatch(d1, d2, k=2) if a.distance < 0.75 * b.distance]
p1 = np.float32([k1[m.queryIdx].pt for m in good])
p2 = np.float32([k2[m.trainIdx].pt for m in good])
M, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
n_in = int(mask.sum())
Minv = np.linalg.inv(M)
corners = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]]).reshape(-1, 1, 2)
view = cv2.perspectiveTransform(corners, M).reshape(-1, 2)
print('特征匹配 good=%d  RANSAC 内点=%d' % (len(good), n_in))
print('照片四角 -> 源图坐标: ' + ', '.join('(%.0f,%.0f)' % (x, y) for x, y in view))
area = abs(cv2.contourArea(view.reshape(-1, 1, 2).astype(np.float32)))
print('视野覆盖源图面积 = %.1f%%' % (100.0 * area / (W * H)))

# ---------- 2. GT 投影 ----------
gt_src = []
for line in open(SRC_LBL):
    q = line.split()
    if len(q) < 5:
        continue
    cx, cy, w, h = map(float, q[1:5])
    gt_src.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
print('源图 GT 总数 = %d' % len(gt_src))

gt = []
for b in gt_src:
    poly = np.float32([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]).reshape(-1, 1, 2)
    w4 = cv2.perspectiveTransform(poly, Minv).reshape(-1, 2)
    x0, y0 = w4[:, 0].min(), w4[:, 1].min()
    x1, y1 = w4[:, 0].max(), w4[:, 1].max()
    cx0, cy0 = max(0.0, x0), max(0.0, y0)
    cx1, cy1 = min(float(pw), x1), min(float(ph), y1)
    if cx1 - cx0 < 3 or cy1 - cy0 < 3:
        continue
    vis = ((cx1 - cx0) * (cy1 - cy0)) / max(1e-6, (x1 - x0) * (y1 - y0))
    if vis < 0.4:
        continue
    gt.append([cx0, cy0, cx1, cy1])
print('投影进照片视野内的 GT = %d' % len(gt))

# ---------- 3. 板端输出 ----------
j = json.load(open(DETS, encoding='utf-8'))
pred = [[d['x'], d['y'], d['x'] + d['w'], d['y'] + d['h']] for d in j['detections']]
confs = [d['conf'] for d in j['detections']]
print('板端输出 = %d 框（json 里 count=%d，fps=%.1f）' % (len(pred), j['count'], j['fps']))


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x1 - x0), max(0.0, y1 - y0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# 置信度从高到低贪心匹配，每个 GT 只配一次
order = sorted(range(len(pred)), key=lambda i: -confs[i])
used = [False] * len(gt)
tp = 0
matched_pairs = []
for i in order:
    best, bi = 0.0, -1
    for k, g in enumerate(gt):
        if used[k]:
            continue
        v = iou(pred[i], g)
        if v > best:
            best, bi = v, k
    if best >= IOU_T and bi >= 0:
        used[bi] = True
        tp += 1
        matched_pairs.append((i, bi, best))
fp = len(pred) - tp
fn = len(gt) - tp
P = tp / len(pred) if pred else 0.0
R = tp / len(gt) if gt else 0.0
F1 = 2 * P * R / (P + R) if P + R else 0.0
print()
print('=== 逐框评测（IoU>=%.1f，板端 conf>=0.30）===' % IOU_T)
print('TP=%d  FP=%d  FN=%d' % (tp, fp, fn))
print('Precision=%.4f  Recall=%.4f  F1=%.4f' % (P, R, F1))
print('平均匹配 IoU = %.3f' % (np.mean([m[2] for m in matched_pairs]) if matched_pairs else 0))

# ---------- 4. 可视化 ----------
vis = photo.copy()
for g in gt:
    cv2.rectangle(vis, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])), (0, 255, 0), 1)
for i, p in enumerate(pred):
    col = (0, 0, 255)
    cv2.rectangle(vis, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), col, 1)
cv2.putText(vis, 'GT=%d(green) PRED=%d(red) TP=%d P=%.2f R=%.2f' % (len(gt), len(pred), tp, P, R),
            (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
vp = os.path.join(OUT, 'e2e_cap0005_overlay.jpg')
cv2.imwrite(vp, vis)
print('可视化 ->', vp)

json.dump({'photo': PHOTO, 'dets': DETS, 'sift_good': len(good), 'sift_inliers': n_in,
           'view_corners_in_src': view.tolist(), 'area_frac': area / (W * H),
           'gt_in_view': len(gt), 'pred': len(pred), 'tp': tp, 'fp': fp, 'fn': fn,
           'precision': P, 'recall': R, 'f1': F1, 'iou_thr': IOU_T,
           'mean_matched_iou': float(np.mean([m[2] for m in matched_pairs])) if matched_pairs else 0.0},
          open(os.path.join(OUT, 'e2e_cap0005.json'), 'w', encoding='utf-8'), indent=1)
print('结果 ->', os.path.join(OUT, 'e2e_cap0005.json'))
