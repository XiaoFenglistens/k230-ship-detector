# SPDX-FileCopyrightText: 2026 XiaoFenglistens
# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""拍屏端到端评测（用自动连拍的 a 图，已证实与推理帧对齐到 ~1px）。

对每一对：
  1) a 图与源图做 SIFT+RANSAC 单应性，把源图 GT 投影到 a 图坐标系
  2) 板端该帧的框（json）与投影后的 GT 按 IoU>=0.5 贪心匹配
  3) 汇总 P / R / F1 / 平均匹配 IoU

用法：python tools\e2e_auto_gt.py
"""
import os as _os
KROOT = _os.environ.get("K230_ROOT", ".")  # 项目根目录，默认当前目录
import glob
import json
import os

import cv2
import numpy as np

AUTO_DIR = _os.path.join(KROOT, 'tmp', 'auto')
SRC_IMG = _os.path.join(KROOT, 'datasets', 'sodaa_ship_hbb', 'images', 'val', '00063__800__0___0.jpg')
SRC_LBL = _os.path.join(KROOT, 'datasets', 'sodaa_ship_hbb', 'labels', 'val', '00063__800__0___0.txt')
OUT = _os.path.join(KROOT, 'runs', 'capture_e2e')
IOU_THR = 0.5
VIS_THR = 0.4


def load_gt(img_path, lbl_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    gts = []
    with open(lbl_path) as f:
        for line in f:
            q = line.split()
            if len(q) < 5:
                continue
            cx, cy, bw, bh = map(float, q[1:5])
            gts.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                        (cx + bw / 2) * w, (cy + bh / 2) * h])
    return img, np.array(gts, dtype=np.float32)


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    iw = np.maximum(ix2 - ix1, 0)
    ih = np.maximum(iy2 - iy1, 0)
    inter = iw * ih
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-6)


def project_gt(gts, H_inv, w, h):
    out = []
    for g in gts:
        pts = np.array([[[g[0], g[1]], [g[2], g[1]], [g[2], g[3]], [g[0], g[3]]]],
                       dtype=np.float32)
        p = cv2.perspectiveTransform(pts, H_inv)[0]
        x1, y1 = p[:, 0].min(), p[:, 1].min()
        x2, y2 = p[:, 0].max(), p[:, 1].max()
        # 可见比例（裁剪到画面内）
        cx1, cy1 = max(x1, 0), max(y1, 0)
        cx2, cy2 = min(x2, w), min(y2, h)
        area = max((x2 - x1) * (y2 - y1), 1e-6)
        vis = max((cx2 - cx1) * (cy2 - cy1), 0) / area
        if vis >= VIS_THR:
            out.append([cx1, cy1, cx2, cy2])
    return np.array(out, dtype=np.float32)


def match(preds, gts, thr=IOU_THR):
    order = np.argsort(-preds[:, 4]) if len(preds) else []
    used = np.zeros(len(gts), dtype=bool)
    tp = 0
    ious = []
    for i in order:
        if len(gts) == 0:
            break
        m = iou_matrix(preds[i:i + 1, :4], gts)[0]
        j = int(np.argmax(m))
        if m[j] >= thr and not used[j]:
            used[j] = True
            tp += 1
            ious.append(float(m[j]))
    fp = len(preds) - tp
    fn = int((~used).sum())
    return tp, fp, fn, ious


def main():
    os.makedirs(OUT, exist_ok=True)
    src, gts = load_gt(SRC_IMG, SRC_LBL)
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=6000)
    kp1, des1 = sift.detectAndCompute(src_gray, None)
    bf = cv2.BFMatcher()

    rows = []
    tot = [0, 0, 0]
    all_ious = []

    for jp in sorted(glob.glob(os.path.join(AUTO_DIR, 'AUTO_*.json'))):
        tag = os.path.splitext(os.path.basename(jp))[0]
        ap = os.path.join(AUTO_DIR, tag + '_a.jpg')
        if not os.path.exists(ap):
            continue
        with open(jp, encoding='utf-8') as f:
            j = json.load(f)
        preds = np.array([[d['x'], d['y'], d['x'] + d['w'], d['y'] + d['h'], d['conf']]
                          for d in j['detections']], dtype=np.float32)
        if len(preds) == 0:
            continue

        ph = cv2.imread(ap)
        ph_gray = cv2.cvtColor(ph, cv2.COLOR_BGR2GRAY)
        kp2, des2 = sift.detectAndCompute(ph_gray, None)
        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) < 12:
            print('%s SIFT 匹配不足 (%d)' % (tag, len(good)))
            continue
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        inliers = int(mask.sum()) if mask is not None else 0
        H_inv = np.linalg.inv(H)

        h, w = ph.shape[:2]
        gt_proj = project_gt(gts, H_inv, w, h)
        tp, fp, fn, ious = match(preds, gt_proj)
        tot[0] += tp
        tot[1] += fp
        tot[2] += fn
        all_ious.extend(ious)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        rows.append({'tag': tag, 'inliers': inliers, 'gt_in_view': int(len(gt_proj)),
                     'pred': int(len(preds)), 'tp': tp, 'fp': fp, 'fn': fn,
                     'precision': round(prec, 4), 'recall': round(rec, 4),
                     'mean_iou': round(float(np.mean(ious)) if ious else 0.0, 4)})
        print('%s inl=%d gt_in_view=%d pred=%d  TP=%d FP=%d FN=%d  P=%.3f R=%.3f IoU=%.3f'
              % (tag, inliers, len(gt_proj), len(preds), tp, fp, fn, prec, rec,
                 float(np.mean(ious)) if ious else 0.0))

    tp, fp, fn = tot
    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    F1 = 2 * P * R / max(P + R, 1e-9)
    print('\n=== 合计 ===')
    print('TP=%d FP=%d FN=%d  P=%.4f R=%.4f F1=%.4f  平均匹配 IoU=%.4f'
          % (tp, fp, fn, P, R, F1, float(np.mean(all_ious)) if all_ious else 0))
    out = {'summary': {'tp': tp, 'fp': fp, 'fn': fn, 'precision': round(P, 4),
                       'recall': round(R, 4), 'f1': round(F1, 4),
                       'mean_iou': round(float(np.mean(all_ious)) if all_ious else 0.0, 4)},
           'pairs': rows}
    with open(os.path.join(OUT, 'auto_gt_eval.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('-> %s' % os.path.join(OUT, 'auto_gt_eval.json'))


if __name__ == '__main__':
    main()
