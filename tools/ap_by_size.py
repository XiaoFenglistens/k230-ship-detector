# -*- coding: utf-8 -*-
"""
按目标尺寸分桶算 AP50 —— 回答「我的模型在哪类目标上强、哪类弱」

buckets（按 800x800 原图上的等效边长 sqrt(w*h) 分）：
  tiny   < 8 px
  small  8-16 px
  medium 16-32 px
  large  > 32 px

输出 runs/ap_by_size_416.json / _640.json
"""
import json
import os
import sys

import numpy as np
import torch
from ultralytics import YOLO

VAL_IMG = r"D:\rs-yolo\datasets\sodaa_ship_hbb\images\val"
VAL_LBL = r"D:\rs-yolo\datasets\sodaa_ship_hbb\labels\val"
OUT_DIR = r"D:\rs-yolo\runs"

BUCKETS = [("tiny", 0, 8), ("small", 8, 16), ("medium", 16, 32), ("large", 32, 1e9)]
CONF = 0.001          # AP 需要完整 PR 曲线，置信度放到很低
IOU_T = 0.5


def load_gt(path, W, H):
    out = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue
            cx, cy, w, h = (float(v) for v in p[1:5])
            x1 = (cx - w / 2) * W
            y1 = (cy - h / 2) * H
            x2 = (cx + w / 2) * W
            y2 = (cy + h / 2) * H
            out.append([x1, y1, x2, y2])
    return np.array(out, dtype=np.float32).reshape(-1, 4)


def iou_mat(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    bb = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def ap_from_scores(scores, tps, n_gt):
    if n_gt == 0:
        return None
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores)
    tps = tps[order]
    ctp = np.cumsum(tps)
    cfp = np.cumsum(1 - tps)
    rec = ctp / (n_gt + 1e-9)
    prec = ctp / (ctp + cfp + 1e-9)
    # 101 点插值
    mrec = np.linspace(0, 1, 101)
    mpre = np.array([prec[rec >= t].max() if np.any(rec >= t) else 0.0 for t in mrec])
    return float(np.trapz(mpre, mrec))


def run(weights, imgsz, tag):
    model = YOLO(weights)
    names = sorted(os.listdir(VAL_IMG))
    scores = {b[0]: [] for b in BUCKETS}
    tps = {b[0]: [] for b in BUCKETS}
    ngt = {b[0]: 0 for b in BUCKETS}

    for i, fn in enumerate(names):
        path = os.path.join(VAL_IMG, fn)
        from PIL import Image
        W, H = Image.open(path).size
        gt = load_gt(os.path.join(VAL_LBL, fn[:-4] + ".txt"), W, H)

        r = model.predict(path, imgsz=imgsz, conf=CONF, iou=0.45, max_det=1000,
                          verbose=False, device=0)[0]
        pred = r.boxes.xyxy.cpu().numpy().astype(np.float32).reshape(-1, 4)
        sc = r.boxes.conf.cpu().numpy().astype(np.float32).reshape(-1)

        # 逐 GT 分桶
        gt_sizes = np.sqrt(np.clip(gt[:, 2] - gt[:, 0], 0, None) * np.clip(gt[:, 3] - gt[:, 1], 0, None))
        gt_bucket = np.full(len(gt), -1, dtype=np.int32)
        for bi, (_, lo, hi) in enumerate(BUCKETS):
            m = (gt_sizes >= lo) & (gt_sizes < hi)
            gt_bucket[m] = bi

        M = iou_mat(pred, gt)
        used = np.zeros(len(gt), dtype=bool)
        for bi, (bname, _, _) in enumerate(BUCKETS):
            ngt[bname] += int((gt_bucket == bi).sum())

        for pi in range(len(pred)):
            if len(gt) == 0:
                # 无 GT 的图，预测全算 FP；归到 large 桶以免污染小目标统计
                scores["large"].append(float(sc[pi]))
                tps["large"].append(0.0)
                continue
            j = int(np.argmax(M[pi]))
            hit = M[pi, j] >= IOU_T and not used[j]
            bi = gt_bucket[j]
            bname = BUCKETS[bi][0] if bi >= 0 else "large"
            scores[bname].append(float(sc[pi]))
            if hit:
                used[j] = True
                tps[bname].append(1.0)
            else:
                tps[bname].append(0.0)

        if (i + 1) % 100 == 0:
            print("  %s  %d/%d" % (tag, i + 1, len(names)), flush=True)

    res = {"imgsz": imgsz, "images": len(names), "iou": IOU_T, "buckets": {}}
    for name, lo, hi in BUCKETS:
        a = ap_from_scores(np.array(scores[name]), np.array(tps[name]), ngt[name])
        res["buckets"][name] = {
            "range_px": [lo, None if hi > 1e8 else hi],
            "n_gt": ngt[name],
            "n_pred": len(scores[name]),
            "ap50": None if a is None else round(a, 4),
        }
    with open(os.path.join(OUT_DIR, "ap_by_size_%s.json" % tag), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    return res


if __name__ == "__main__":
    jobs = [
        (r"D:\rs-yolo\runs\ship_416_v1\weights\best.pt", 416, "416"),
        (r"D:\rs-yolo\runs\ship_640_v1\weights\best.pt", 640, "640"),
    ]
    allres = {}
    for w, s, tag in jobs:
        print("=== %s ===" % tag, flush=True)
        allres[tag] = run(w, s, tag)
    print(json.dumps(allres, ensure_ascii=False, indent=1))
