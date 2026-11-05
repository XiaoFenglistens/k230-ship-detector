# SPDX-FileCopyrightText: 2026 XiaoFenglistens
# SPDX-License-Identifier: MIT
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
失败案例分析：把 416 模型在 val 上的漏检(FN)与误检(FP)分类、计数、切成小图拼成总览，
用于人工归纳"模型在哪几类场景会错"。

用法：
    python tools\failure_analysis.py --n 200
"""
import os as _os
KROOT = _os.environ.get("K230_ROOT", ".")  # 项目根目录，默认当前目录

import os
import sys
import glob
import json
import argparse

import numpy as np
import cv2
from ultralytics import YOLO

MODEL = _os.path.join(KROOT, 'runs', 'ship_416_v1', 'weights', 'best.pt')
IMG_DIR = _os.path.join(KROOT, 'datasets', 'sodaa_ship_hbb', 'images', 'val')
LBL_DIR = _os.path.join(KROOT, 'datasets', 'sodaa_ship_hbb', 'labels', 'val')
OUT_DIR = _os.path.join(KROOT, 'runs', 'failure_analysis')
SIZE = 416
CONF = 0.30


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
    iw = np.maximum(0.0, np.minimum(ax2, bx2) - np.maximum(ax1, bx1))
    ih = np.maximum(0.0, np.minimum(ay2, by2) - np.maximum(ay1, by1))
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def crop_ctx(img, box, mul=3.0, min_side=110):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max((x2 - x1), (y2 - y1)) * mul
    side = max(side, min_side)
    H, W = img.shape[:2]
    a = int(max(0, cx - side / 2))
    b = int(max(0, cy - side / 2))
    c = int(min(W, cx + side / 2))
    d = int(min(H, cy + side / 2))
    if c - a < 8 or d - b < 8:
        return None
    return img[b:d, a:c].copy()


def sheet(crops, cols, cell, path, title):
    if not crops:
        return
    rows = (len(crops) + cols - 1) // cols
    canvas = np.full((rows * (cell + 18) + 26, cols * cell, 3), 28, np.uint8)
    for i, (c, cap) in enumerate(crops):
        r, cc = divmod(i, cols)
        im = cv2.resize(c, (cell, cell), interpolation=cv2.INTER_NEAREST)
        y0 = 26 + r * (cell + 18)
        canvas[y0:y0 + cell, cc * cell:(cc + 1) * cell] = im
        cv2.putText(canvas, cap, (cc * cell + 3, y0 + cell + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(canvas, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, canvas)
    print("  拼图 ->", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-sheet", type=int, default=24)
    ap.add_argument("--max-det", type=int, default=1000,
                    help="必须 >= 单图最多目标数（本数据集 668），否则密集图会被截断当作漏检")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    rng = np.random.RandomState(args.seed)
    if len(files) > args.n:
        files = [files[i] for i in np.sort(rng.choice(len(files), args.n, replace=False))]
    print("[数据] %d 张" % len(files))

    model = YOLO(MODEL)

    fn_by_size = {"<8": 0, "8-16": 0, "16-32": 0, ">32": 0}
    gt_by_size = {"<8": 0, "8-16": 0, "16-32": 0, ">32": 0}
    fn_total = 0
    gt_total = 0
    fp_total = 0
    fp_near = 0      # 与某个 GT 有 0.1~0.5 重叠：定位偏差 / 重复框
    fp_far = 0       # 与所有 GT 都不沾边：纯误报
    tp_total = 0

    fn_crops, fp_near_crops, fp_far_crops = [], [], []
    per_image = []

    for k, path in enumerate(files):
        img = cv2.imread(path)
        H, W = img.shape[:2]
        name = os.path.basename(path)
        gt = load_gt(os.path.join(LBL_DIR, name[:-4] + ".txt"), W, H)
        gt_total += len(gt)

        res = model.predict(img, imgsz=SIZE, conf=CONF, iou=0.45, max_det=args.max_det, verbose=False)[0]
        pb = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.zeros((0, 4), np.float32)
        ps = res.boxes.conf.cpu().numpy() if len(res.boxes) else np.zeros((0,), np.float32)

        if len(gt) and len(pb):
            m = iou_matrix(pb, gt)
            tp_total += int((m.max(axis=1) >= 0.5).sum())
            matched_gt = m.max(axis=0) >= 0.5
            fn_mask = ~matched_gt
        elif len(gt):
            tp_total += 0
            fn_mask = np.ones(len(gt), bool)
        else:
            fn_mask = np.zeros(0, bool)

        n_fn = int(fn_mask.sum())
        fn_total += n_fn
        # 全部 GT 按尺寸分桶（用于算每个桶的召回率）
        if len(gt):
            for b in gt:
                s = float(np.sqrt(max(0.0, (b[2] - b[0]) * (b[3] - b[1]))))
                if s < 8:
                    gt_by_size["<8"] += 1
                elif s < 16:
                    gt_by_size["8-16"] += 1
                elif s < 32:
                    gt_by_size["16-32"] += 1
                else:
                    gt_by_size[">32"] += 1
        # FN 按目标尺寸分桶
        for b in gt[fn_mask] if len(gt) else []:
            s = float(np.sqrt(max(0.0, (b[2] - b[0]) * (b[3] - b[1]))))
            if s < 8:
                fn_by_size["<8"] += 1
            elif s < 16:
                fn_by_size["8-16"] += 1
            elif s < 32:
                fn_by_size["16-32"] += 1
            else:
                fn_by_size[">32"] += 1

        # FP 分类：与所有 GT 的最大重叠
        if len(pb):
            if len(gt):
                mg = iou_matrix(pb, gt).max(axis=1)
            else:
                mg = np.zeros(len(pb), np.float32)
            keep_fp = mg < 0.5
            for i in np.where(keep_fp)[0]:
                fp_total += 1
                near = mg[i] >= 0.1
                if near:
                    fp_near += 1
                else:
                    fp_far += 1
                c = crop_ctx(img, pb[i])
                if c is None:
                    continue
                cap = "%s c%.2f iou%.2f" % (name[:12], ps[i], mg[i])
                (fp_near_crops if near else fp_far_crops).append((c, cap))

        for b in (gt[fn_mask] if len(gt) else []):
            c = crop_ctx(img, b)
            if c is None:
                continue
            s = float(np.sqrt(max(0.0, (b[2] - b[0]) * (b[3] - b[1]))))
            fn_crops.append((c, "%s %.0fpx" % (name[:12], s)))

        per_image.append({"name": name, "gt": int(len(gt)), "pred": int(len(pb)),
                          "tp": int((iou_matrix(pb, gt).max(axis=1) >= 0.5).sum()) if (len(pb) and len(gt)) else 0,
                          "fn": n_fn})
        if (k + 1) % 40 == 0:
            print("  %d/%d" % (k + 1, len(files)), flush=True)

    stats = {
        "n_images": len(files),
        "gt_total": gt_total,
        "tp": tp_total,
        "fn_total": fn_total,
        "recall": tp_total / (gt_total + 1e-9),
        "fp_total": fp_total,
        "fp_near_gt": fp_near,
        "fp_far_from_gt": fp_far,
        "precision_est": tp_total / (tp_total + fp_total + 1e-9),
        "fn_by_size": fn_by_size,
        "fn_share_by_size": {k: v / (fn_total + 1e-9) for k, v in fn_by_size.items()},
        "gt_by_size": gt_by_size,
        "recall_by_size": {k: 1.0 - fn_by_size[k] / (gt_by_size[k] + 1e-9) for k in gt_by_size},
    }
    print("\n===== 统计 =====")
    for k, v in stats.items():
        print("  %-18s %s" % (k, v))

    # ---- 画汇总图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        for cand in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
            try:
                font_manager.findfont(font_manager.FontProperties(family=cand), fallback_to_default=False)
                plt.rcParams["font.family"] = cand
                break
            except Exception:
                continue
        plt.rcParams["axes.unicode_minus"] = False

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        buckets = ["<8", "8-16", "16-32", ">32"]
        rec = [stats["recall_by_size"][b] * 100 for b in buckets]
        gtn = [stats["gt_by_size"][b] for b in buckets]
        bars = axes[0].bar(buckets, rec, color=["#c0392b", "#e67e22", "#2e9e5b", "#3b7dd8"])
        axes[0].set_ylim(0, 100)
        axes[0].set_ylabel("召回率 (%)")
        axes[0].set_xlabel("目标等效边长 (px)")
        axes[0].set_title("各尺寸目标的召回率（416 输入，conf=0.30）")
        for b, v, n in zip(bars, rec, gtn):
            axes[0].text(b.get_x() + b.get_width() / 2, v + 1.5,
                         "%.1f%%\n(n=%d)" % (v, n), ha="center", fontsize=9)

        parts = [stats["tp"], stats["fp_near_gt"], stats["fp_far_from_gt"]]
        names = ["TP 正确", "FP 贴近船\n(重复/大框)", "FP 远离船\n(纯误报)"]
        axes[1].bar(names, parts, color=["#2e9e5b", "#e67e22", "#c0392b"])
        axes[1].set_ylim(0, max(parts) * 1.22)
        axes[1].set_title("检出框的构成（200 张 val 图，共 %d 个预测）" % (sum(parts)))
        axes[1].set_ylabel("框数量")
        for i, v in enumerate(parts):
            axes[1].text(i, v + max(parts) * 0.015, "%d\n%.1f%%" % (v, 100.0 * v / sum(parts)),
                         ha="center", fontsize=9)
        fig.suptitle("失败分析：错在哪里", fontsize=13)
        fig.tight_layout()
        figpath = os.path.join(OUT_DIR, "failure_summary.png")
        fig.savefig(figpath, dpi=110)
        print("  汇总图 ->", figpath)
    except Exception as e:
        print("  画图失败:", e)

    with open(os.path.join(OUT_DIR, "failure_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "per_image": per_image}, f, ensure_ascii=False, indent=2)

    ps_ = args.per_sheet
    # FN 按尺寸从大到小排，便于看"大目标为什么也漏"
    fn_crops.sort(key=lambda t: -float(t[1].split()[-1].rstrip("px")))
    sheet(fn_crops[:ps_], 6, 128, os.path.join(OUT_DIR, "sheet_fn.jpg"),
          "FALSE NEGATIVE (missed ships) - biggest first")
    sheet(fn_crops[-ps_:], 6, 128, os.path.join(OUT_DIR, "sheet_fn_small.jpg"),
          "FALSE NEGATIVE (missed ships) - smallest")
    fp_near_crops.sort(key=lambda t: -float(t[1].split()[1][1:]))
    fp_far_crops.sort(key=lambda t: -float(t[1].split()[1][1:]))

    def samp(lst, k, seed=7):
        """随机抽样，避免只看置信度最高的那一批（那会集中在个别图里，不具代表性）"""
        if len(lst) <= k:
            return lst
        r = np.random.RandomState(seed)
        return [lst[i] for i in np.sort(r.choice(len(lst), k, replace=False))]

    sheet(samp(fp_near_crops, ps_), 6, 128, os.path.join(OUT_DIR, "sheet_fp_near.jpg"),
          "FALSE POSITIVE near a ship (random sample of %d)" % len(fp_near_crops))
    sheet(samp(fp_far_crops, ps_ * 2), 6, 128, os.path.join(OUT_DIR, "sheet_fp_far.jpg"),
          "FALSE POSITIVE far from any ship (random sample of %d)" % len(fp_far_crops))
    print("\n完成，输出目录:", OUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
