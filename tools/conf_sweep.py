# -*- coding: utf-8 -*-
"""
置信度阈值消融（conf sweep）。

问题：板端程序里 CONFIDENCE_THRESHOLD = 0.30 是不是拍的？

做法：
  1) 用很低的阈值（0.01）跑一遍全量 val 图，把**所有**检测框连同置信度留下来；
  2) 再在离线状态下按不同阈值过滤这些框、重新做匹配统计；
     → 一次推理扫完整条曲线，避免每个阈值都重跑一遍模型。
  3) 每个阈值都算 TP/FP/FN（IoU>=0.5 贪心匹配）、precision、recall、F1，
     并统计平均每图框数（这个数直接关系到板端 max_boxes_num）。

匹配口径与 tools/failure_analysis.py 完全一致，保证数字可横向比较。

用法：
  python tools\\conf_sweep.py --model runs\\ship_416_v1\\weights\\best.pt --imgsz 416
  python tools\\conf_sweep.py --model runs\\ship_512_v1\\weights\\best.pt --imgsz 512
"""
import argparse
import json
import os

import numpy as np

ROOT = r"D:\rs-yolo"
VAL_IMG = os.path.join(ROOT, "datasets", "sodaa_ship_hbb", "images", "val")
VAL_LBL = os.path.join(ROOT, "datasets", "sodaa_ship_hbb", "labels", "val")
OUT_DIR = os.path.join(ROOT, "runs")

# 扫描的阈值点
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]
MATCH_IOU = 0.5
LOW_CONF = 0.01          # 第一次推理用的低阈值
MAX_DET = 1000           # 必须 >= 单图最多目标数（本数据集 668）


def load_gt(lbl_path, w, h):
    """读 YOLO txt（cx cy w h，归一化）→ xyxy 绝对坐标"""
    out = []
    if not os.path.exists(lbl_path):
        return np.zeros((0, 4), dtype=np.float32)
    with open(lbl_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            cx, cy, bw, bh = map(float, parts[1:5])
            out.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                        (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(out, dtype=np.float32) if out else np.zeros((0, 4), dtype=np.float32)


def iou_matrix(a, b):
    """a: (n,4) b: (m,4) xyxy -> (n,m)"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)
    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def match_flags(pred, gt, iou_thr=MATCH_IOU):
    """贪心匹配，返回 (tp, fp, fn)。每条预测单独算 IoU 行（曾误用缓存导致算错）。"""
    if len(pred) == 0:
        return 0, 0, len(gt)
    if len(gt) == 0:
        return 0, len(pred), 0
    used = np.zeros(len(gt), dtype=bool)
    tp = 0
    for i in range(len(pred)):
        ious = iou_matrix(pred[i:i + 1], gt)[0]
        ious[used] = -1.0
        j = int(np.argmax(ious))
        if ious[j] >= iou_thr:
            used[j] = True
            tp += 1
    fp = len(pred) - tp
    fn = int((~used).sum())
    return tp, fp, fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--limit", type=int, default=0, help="0 = 全部 val")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from ultralytics import YOLO

    names = sorted(f for f in os.listdir(VAL_IMG)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if args.limit:
        names = names[:args.limit]
    print("图片数 %d，模型 %s，imgsz=%d" % (len(names), args.model, args.imgsz))

    model = YOLO(args.model)

    # ---- 第一遍：低阈值跑一遍，把所有预测存下来 ----
    per_image = []      # [(pred_xyxy, pred_conf, gt_xyxy)]
    n_gt = 0
    for k, nm in enumerate(names):
        img_path = os.path.join(VAL_IMG, nm)
        r = model.predict(img_path, imgsz=args.imgsz, conf=LOW_CONF, iou=0.45,
                          max_det=MAX_DET, verbose=False)[0]
        h, w = r.orig_shape
        gt = load_gt(os.path.join(VAL_LBL, nm[:-4] + ".txt"), w, h)
        pred = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        conf = r.boxes.conf.cpu().numpy().astype(np.float32)
        per_image.append((pred, conf, gt))
        n_gt += len(gt)
        if (k + 1) % 100 == 0:
            print("  推理 %d/%d" % (k + 1, len(names)), flush=True)
    print("GT 总数 %d" % n_gt, flush=True)

    # ---- 第二遍：离线扫阈值 ----
    rows = []
    for th in THRESHOLDS:
        tp = fp = fn = 0
        n_box = 0
        for pred, conf, gt in per_image:
            keep = conf >= th
            p = pred[keep]
            n_box += len(p)
            a, b, c = match_flags(p, gt)
            tp += a
            fp += b
            fn += c
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        rows.append({
            "conf": th,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "boxes_total": n_box,
            "boxes_per_image": round(n_box / len(names), 1),
        })
        print("conf=%.2f  P=%.3f  R=%.3f  F1=%.3f  框/图=%.1f"
              % (th, prec, rec, f1, n_box / len(names)), flush=True)

    best = max(rows, key=lambda r: r["f1"])
    print("\n最佳 F1 阈值 = %.2f（F1=%.3f, P=%.3f, R=%.3f）"
          % (best["conf"], best["f1"], best["precision"], best["recall"]))

    out = args.out or os.path.join(OUT_DIR, "conf_sweep_%d.json" % args.imgsz)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "imgsz": args.imgsz,
            "n_images": len(names),
            "n_gt": n_gt,
            "match_iou": MATCH_IOU,
            "sweep": rows,
            "best_f1": best,
        }, f, ensure_ascii=False, indent=2)
    print("-> %s" % out)


if __name__ == "__main__":
    main()
