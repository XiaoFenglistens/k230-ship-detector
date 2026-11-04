# SPDX-FileCopyrightText: 2026 XiaoFenglistens
# SPDX-License-Identifier: MIT
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
量化误差全量测定：nncase Simulator(kmodel) vs onnxruntime(onnx)
在同一批 val 图上各自算 mAP50 / mAP50-95，给出量化带来的精度损失。

用法：
    python tools\quant_gap.py --n 60
"""

import os
import sys
import glob
import json
import time
import argparse
import importlib.util

_spec = importlib.util.find_spec("nncase")
_pkg = os.path.dirname(os.path.abspath(_spec.origin))
_site = os.path.dirname(_pkg)
os.environ["NNCASE_PLUGIN_PATH"] = _site
if _site not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _site + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import cv2
import nncase
import nncase_kpu  # noqa
import onnxruntime as ort

ROOT = r"D:\rs-yolo\runs\ship_416_v1\weights"
KMODEL = os.path.join(ROOT, "best.kmodel")
ONNX = os.path.join(ROOT, "best.onnx")
IMG_DIR = r"D:\rs-yolo\datasets\sodaa_ship_hbb\images\val"
LBL_DIR = r"D:\rs-yolo\datasets\sodaa_ship_hbb\labels\val"
OUT = r"D:\rs-yolo\runs\quant_gap.json"

SIZE = 416
NC = 1
CONF = 0.001        # AP 用低阈值，才能画出完整 PR 曲线
NMS_IOU = 0.45
MAX_DET = 1000      # 必须 >= 单图最多目标数（本数据集 668），否则密集图被截断


def letterbox(img, size):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (size - nw) / 2.0, (size - nh) / 2.0
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, (left, top)


def nms(boxes, scores, iou_thres):
    if len(boxes) == 0:
        return np.zeros((0,), dtype=int)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thres]
    return np.array(keep, dtype=int)


def decode(out, conf=CONF, iou=NMS_IOU, max_det=MAX_DET):
    """out shape (1, 4+nc, N) -> boxes xyxy (已经是 416 letterbox 坐标系)"""
    p = out[0].transpose(1, 0)
    boxes_xywh = p[:, :4]
    scores = p[:, 4:4 + NC].max(axis=1)
    m = scores >= conf
    if m.sum() == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
    b, s = boxes_xywh[m], scores[m]
    cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    k = nms(boxes, s, iou)[:max_det]
    return boxes[k], s[k]


def load_gt(txt_path, W, H):
    if not os.path.exists(txt_path):
        return np.zeros((0, 4), np.float32)
    out = []
    with open(txt_path) as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue
            cx, cy, w, h = [float(v) for v in p[1:5]]
            out.append([(cx - w / 2) * W, (cy - h / 2) * H,
                        (cx + w / 2) * W, (cy + h / 2) * H])
    return np.array(out, np.float32) if out else np.zeros((0, 4), np.float32)


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    iw = np.minimum(ax2, bx2) - np.maximum(ax1, bx1)
    ih = np.minimum(ay2, by2) - np.maximum(ay1, by1)
    iw = np.maximum(0.0, iw)
    ih = np.maximum(0.0, ih)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def average_precision(tp_flags, n_gt):
    if n_gt == 0:
        return float("nan")
    tp = np.cumsum(tp_flags)
    fp = np.cumsum(~np.asarray(tp_flags))
    recall = tp / (n_gt + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.linspace(0, 1, 101)
    return float(np.mean(np.interp(idx, mrec, mpre)))


def eval_preds(all_preds, gts, n_gt, iou_thr):
    """all_preds: [(conf, img_idx, box_xyxy)]"""
    order = sorted(all_preds, key=lambda t: -t[0])
    used = [np.zeros(len(g), bool) for g in gts]
    tp_flags = []
    for conf, idx, box in order:
        g = gts[idx]
        if len(g) == 0:
            tp_flags.append(False)
            continue
        # 每条预测的框不同，必须逐条算 IoU（曾误用缓存，导致 mAP 算错）
        row = iou_matrix(box.reshape(1, 4), g)[0].copy()
        row[used[idx]] = -1.0
        j = int(row.argmax())
        if row[j] >= iou_thr:
            used[idx][j] = True
            tp_flags.append(True)
        else:
            tp_flags.append(False)
    return average_precision(tp_flags, n_gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kmodel", default=KMODEL)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    rng = np.random.RandomState(args.seed)
    if len(files) > args.n:
        idxs = np.sort(rng.choice(len(files), args.n, replace=False))
        files = [files[i] for i in idxs]
    print("[数据] %d 张 val 图" % len(files), flush=True)

    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    sim = nncase.Simulator()
    with open(args.kmodel, "rb") as f:
        sim.load_model(f.read())
    print("[模型] %s" % os.path.basename(args.kmodel), flush=True)
    print("[模型] onnx + kmodel 已加载", flush=True)

    gts, preds_onnx, preds_km = [], [], []
    n_gt = 0
    t0 = time.time()
    for i, path in enumerate(files):
        bgr = cv2.imread(path)
        H, W = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        lb, r, (padl, padt) = letterbox(rgb, SIZE)

        gt = load_gt(os.path.join(LBL_DIR, os.path.basename(path)[:-4] + ".txt"), W, H)
        gts.append(gt)
        n_gt += len(gt)

        x_u8 = np.transpose(lb, (2, 0, 1))[None].astype(np.uint8)
        x_f32 = x_u8.astype(np.float32) / 255.0

        o_onnx = sess.run(None, {in_name: x_f32})[0]
        sim.set_input_tensor(0, nncase.RuntimeTensor.from_numpy(x_u8))
        sim.run()
        o_km = sim.get_output_tensor(0).to_numpy()

        for out, bag in ((o_onnx, preds_onnx), (o_km, preds_km)):
            boxes, scores = decode(out)
            if len(boxes):
                boxes = boxes.copy()
                boxes[:, [0, 2]] = (boxes[:, [0, 2]] - padl) / r
                boxes[:, [1, 3]] = (boxes[:, [1, 3]] - padt) / r
            for b, s in zip(boxes, scores):
                bag.append((float(s), i, b))

        if (i + 1) % 5 == 0 or i == len(files) - 1:
            el = time.time() - t0
            msg = "  %d/%d  %.0fs  (%.1fs/张)  预测数 onnx=%d km=%d" % (
                i + 1, len(files), el, el / (i + 1), len(preds_onnx), len(preds_km))
            print(msg, flush=True)
            try:
                with open(args.out + ".progress", "w", encoding="utf-8") as pf:
                    pf.write("%d/%d  elapsed=%.0fs  %.1fs/img\n"
                             % (i + 1, len(files), el, el / (i + 1)))
            except Exception:
                pass

    thrs = [round(0.5 + 0.05 * k, 2) for k in range(10)]
    res = {}
    for name, bag in (("onnx", preds_onnx), ("kmodel", preds_km)):
        ap50 = eval_preds(bag, gts, n_gt, 0.5)
        aps = [eval_preds(bag, gts, n_gt, t) for t in thrs]
        res[name] = {
            "n_images": len(files),
            "n_gt": int(n_gt),
            "n_pred": len(bag),
            "mAP50": ap50,
            "mAP50_95": float(np.mean(aps)),
            "ap_by_thr": dict(zip([str(t) for t in thrs], aps)),
        }
        print("\n===== %s =====" % name)
        print("  mAP50    = %.4f" % ap50)
        print("  mAP50-95 = %.4f" % float(np.mean(aps)))

    print("\n===== 量化损失（kmodel - onnx）=====")
    print("  d_mAP50    = %+.4f" % (res["kmodel"]["mAP50"] - res["onnx"]["mAP50"]))
    print("  d_mAP50-95 = %+.4f" % (res["kmodel"]["mAP50_95"] - res["onnx"]["mAP50_95"]))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    res["kmodel_path"] = args.kmodel
    print("\n结果已写入 %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
