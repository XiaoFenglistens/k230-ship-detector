# SPDX-FileCopyrightText: 2026 XiaoFenglistens
# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""分析自动连拍：a→b 的板子位移 与 框相对 a/b 的偏移。

三个量：
1. shift_ab  —— 同一台板子前后两张干净图之间的位移（纯机械抖动，与模型无关）
2. board vs a —— 板端框与 a 上 PC 检测框的最佳整体平移（越接近 (0,0) 越说明映射没问题）
3. board vs b —— 同上，但用 b

用法：python tools\e2e_auto_analyze.py
"""
import glob
import json
import os

import cv2
import numpy as np
from ultralytics import YOLO

AUTO_DIR = r'D:\rs-yolo\tmp\auto'
MODEL = r'D:\rs-yolo\runs\ship_512_v1\weights\best.pt'
OUT = r'D:\rs-yolo\runs\capture_e2e'
IMGSZ = 512
CONF = 0.30
NMS_IOU = 0.45


def load_boxes(path):
    with open(path, encoding='utf-8') as f:
        j = json.load(f)
    out = []
    for d in j['detections']:
        out.append([d['x'], d['y'], d['x'] + d['w'], d['y'] + d['h']])
    return j, np.array(out, dtype=np.float32)


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)
    iw = np.maximum(ix2 - ix1, 0)
    ih = np.maximum(iy2 - iy1, 0)
    inter = iw * ih
    ua = ((ax2 - ax1) * (ay2 - ay1)) + ((bx2 - bx1) * (by2 - by1)) - inter
    return inter / np.maximum(ua, 1e-6)


def best_shift(boxes, ref, rng=40, step=2):
    """网格搜索整体平移量，返回 (dx, dy, mean_iou)。dx/dy 是加在 boxes 上的偏移。"""
    best = (0, 0, -1.0)
    for dy in range(-rng, rng + 1, step):
        for dx in range(-rng, rng + 1, step):
            moved = boxes + np.array([dx, dy, dx, dy], dtype=np.float32)
            m = iou_matrix(moved, ref)
            score = m.max(axis=1).mean() if m.size else 0.0
            if score > best[2]:
                best = (dx, dy, score)
    # 细化（步长 1）
    dx0, dy0, _ = best
    for dy in range(dy0 - 2, dy0 + 3):
        for dx in range(dx0 - 2, dx0 + 3):
            moved = boxes + np.array([dx, dy, dx, dy], dtype=np.float32)
            m = iou_matrix(moved, ref)
            score = m.max(axis=1).mean() if m.size else 0.0
            if score > best[2]:
                best = (dx, dy, score)
    return best


def zero_shift_iou(boxes, ref):
    m = iou_matrix(boxes, ref)
    if m.size == 0:
        return 0.0
    return float(m.max(axis=1).mean())


def main():
    os.makedirs(OUT, exist_ok=True)
    model = YOLO(MODEL)
    rows = []

    for json_path in sorted(glob.glob(os.path.join(AUTO_DIR, 'AUTO_*.json'))):
        tag = os.path.splitext(os.path.basename(json_path))[0]
        pa = os.path.join(AUTO_DIR, tag + '_a.jpg')
        pb = os.path.join(AUTO_DIR, tag + '_b.jpg')
        if not (os.path.exists(pa) and os.path.exists(pb)):
            continue
        j, boxes = load_boxes(json_path)

        ga = cv2.imread(pa, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        gb = cv2.imread(pb, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        win = cv2.createHanningWindow((ga.shape[1], ga.shape[0]), cv2.CV_32F)
        (sx, sy), resp = cv2.phaseCorrelate(ga, gb, win)

        ia = cv2.imread(pa)
        ib = cv2.imread(pb)
        ra = model.predict(ia, imgsz=IMGSZ, conf=CONF, iou=NMS_IOU,
                           max_det=1000, verbose=False)[0]
        rb = model.predict(ib, imgsz=IMGSZ, conf=CONF, iou=NMS_IOU,
                           max_det=1000, verbose=False)[0]
        pca = ra.boxes.xyxy.cpu().numpy()
        pcb = rb.boxes.xyxy.cpu().numpy()

        z_a = zero_shift_iou(boxes, pca)
        z_b = zero_shift_iou(boxes, pcb)
        sh_a = best_shift(boxes, pca)
        sh_b = best_shift(boxes, pcb)

        rows.append({
            'tag': tag, 'board': len(boxes), 'pc_a': len(pca), 'pc_b': len(pcb),
            'shift_ab': (round(float(sx), 2), round(float(sy), 2)),
            'ab_resp': round(float(resp), 3),
            'zero_iou_a': round(z_a, 3), 'zero_iou_b': round(z_b, 3),
            'shift_a': sh_a, 'shift_b': sh_b,
        })
        print('%s board=%d pc_a=%d pc_b=%d  a->b=(%+.1f,%+.1f) resp=%.2f  '
              'zeroIoU a=%.3f b=%.3f  bestShift a=(%+d,%+d) %.3f  b=(%+d,%+d) %.3f'
              % (tag, len(boxes), len(pca), len(pcb), sx, sy, resp,
                 z_a, z_b, sh_a[0], sh_a[1], sh_a[2], sh_b[0], sh_b[1], sh_b[2]))

    out_json = os.path.join(OUT, 'auto_pairs.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print('\n-> %s' % out_json)


if __name__ == '__main__':
    main()
