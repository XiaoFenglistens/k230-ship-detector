# SPDX-FileCopyrightText: 2026 XiaoFenglistens
# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""
B 实验：拍摄条件 -> 检出率曲线

目的：回答「摄像头离屏幕多远、船还剩多少像素，模型还能检出多少」。
方法：把 val 集图片按不同「有效像素保留比例 k」降质（降采样再升采样 + 轻微高斯模糊 +
      JPEG 重压缩），模拟摄像头拍屏幕时目标像素变少的效果，再用 best.pt 推理，
      与 GT 按 IoU>=0.5 匹配，统计 Recall / Precision / F1。

用法：
    python tools/capture_experiment.py --n 200
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from PIL import Image, ImageFilter
from ultralytics import YOLO

IMG_DIR = r"D:\rs-yolo\datasets\sodaa_ship_hbb\images\val"
LBL_DIR = r"D:\rs-yolo\datasets\sodaa_ship_hbb\labels\val"
WEIGHTS = r"D:\rs-yolo\runs\ship_416_v1\weights\best.pt"
OUT = r"D:\rs-yolo\runs\capture_experiment.json"

# 有效像素保留比例 k（1.0 = 原图直送；0.5 = 目标像素只剩一半）
LEVELS = [1.0, 0.75, 0.55, 0.40, 0.28, 0.18]
CONF = 0.30
IOU_MATCH = 0.5


def load_gt(lbl_path, W, H):
    """读到 (x1,y1,x2,y2) 的绝对像素 GT 框。"""
    gt = []
    if not os.path.exists(lbl_path):
        return gt
    with open(lbl_path) as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue
            cx, cy, w, h = map(float, p[1:5])
            gt.append([(cx - w / 2) * W, (cy - h / 2) * H,
                       (cx + w / 2) * W, (cy + h / 2) * H])
    return np.array(gt, dtype=np.float32) if gt else np.zeros((0, 4), np.float32)


def degrade(img, k):
    """模拟摄像头拍屏幕：降采样 -> 升采样 -> 轻微模糊 -> JPEG。"""
    if k >= 0.999:
        return img
    W, H = img.size
    sw, sh = max(1, int(W * k)), max(1, int(H * k))
    small = img.resize((sw, sh), Image.BILINEAR)
    back = small.resize((W, H), Image.BILINEAR)
    # 屏幕拍摄还有轻微失焦/摩尔纹，用一点模糊近似
    back = back.filter(ImageFilter.GaussianBlur(radius=max(0.4, 1.2 * (1 - k))))
    return back


def iou_matrix(a, b):
    """a: (N,4) xyxy, b: (M,4) xyxy -> (N,M)"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    iw = np.maximum(0, np.minimum(a[:, 2][:, None], b[:, 2][None, :])
                    - np.maximum(a[:, 0][:, None], b[:, 0][None, :]))
    ih = np.maximum(0, np.minimum(a[:, 3][:, None], b[:, 3][None, :])
                    - np.maximum(a[:, 1][:, None], b[:, 1][None, :]))
    inter = iw * ih
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    ua = aa[:, None] + bb[None, :] - inter
    return inter / np.maximum(ua, 1e-6)


def evaluate_level(model, files, k, tmp_path):
    """返回 dict(tp, fp, fn, recall, precision)"""
    tp = fp = fn = 0
    for name in files:
        img = Image.open(os.path.join(IMG_DIR, name)).convert("RGB")
        W, H = img.size
        dg = degrade(img, k)
        dg.save(tmp_path, quality=85)   # 走一次 JPEG，模拟重新编码
        res = model.predict(tmp_path, imgsz=416, conf=CONF, iou=0.45,
                            max_det=300, verbose=False, device=0)[0]
        if res.boxes is not None and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy()
        else:
            xyxy = np.zeros((0, 4), np.float32)
        gt = load_gt(os.path.join(LBL_DIR, name[:-4] + ".txt"), W, H)
        if len(gt) == 0:
            fp += len(xyxy)
            continue
        if len(xyxy) == 0:
            fn += len(gt)
            continue
        iou = iou_matrix(xyxy, gt)
        used_gt = np.zeros(len(gt), bool)
        for i in range(len(xyxy)):
            j = int(np.argmax(iou[i]))
            if iou[i, j] >= IOU_MATCH and not used_gt[j]:
                used_gt[j] = True
                tp += 1
            else:
                fp += 1
        fn += int((~used_gt).sum())
    rec = tp / max(tp + fn, 1)
    pre = tp / max(tp + fp, 1)
    f1 = 2 * rec * pre / max(rec + pre, 1e-6)
    return dict(k=k, tp=tp, fp=fp, fn=fn, recall=round(rec, 4),
                precision=round(pre, 4), f1=round(f1, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(IMG_DIR)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    random.Random(args.seed).shuffle(files)
    files = files[:args.n]
    print("样本数", len(files), "  每级都要跑一遍推理")

    torch.cuda.set_per_process_memory_fraction(0.85)
    model = YOLO(WEIGHTS)
    tmp = os.path.join(os.environ.get("TEMP", r"D:\rs-yolo\tmp"), "_cap_tmp.jpg")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)

    rows = []
    for k in LEVELS:
        r = evaluate_level(model, files, k, tmp)
        rows.append(r)
        print("k=%.2f  recall=%.4f  precision=%.4f  f1=%.4f  (tp=%d fp=%d fn=%d)"
              % (k, r["recall"], r["precision"], r["f1"], r["tp"], r["fp"], r["fn"]))

    # 换算成「中位目标像素」帮助理解
    med_px = 22.0     # 数据集中位目标约 22px（@800）
    for r in rows:
        r["median_target_px"] = round(med_px * r["k"], 1)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(dict(samples=len(files), conf=CONF, iou_match=IOU_MATCH, rows=rows),
                  f, ensure_ascii=False, indent=2)
    print("\n写入", OUT)


if __name__ == "__main__":
    main()
