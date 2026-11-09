# SPDX-FileCopyrightText: 2026 XiaoFenglistens
# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""
跨域评测：拿一个 COCO 标注的外部数据集，评估我们训好的船舶检测模型。

用法：
  python tools\\eval_crossdomain.py --name hrsid_sar ^
      --images "D:\\rs-yolo\\datasets\\crossdomain\\hrsid_src\\HRSID_JPG\\JPEGImages" ^
      --coco   "D:\\rs-yolo\\datasets\\crossdomain\\hrsid_src\\HRSID_JPG\\annotations\\test2017.json" ^
      --model  "D:\\rs-yolo\\runs\\ship_416_v1\\weights\\best.pt" ^
      --imgsz 416 --limit 300

输出：runs\crossdomain\<name>.json（mAP50 / mAP50-95 / P / R / 每张统计）+ 若干可视化
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np


# ---------------- IoU / 匹配 / AP ----------------

def iou_matrix(a, b):
    """a: (N,4) xyxy, b: (M,4) xyxy -> (N,M)"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    area_a = np.clip(ax2 - ax1, 0, None) * np.clip(ay2 - ay1, 0, None)
    area_b = np.clip(bx2 - bx1, 0, None) * np.clip(by2 - by1, 0, None)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def match_flags(preds, gts, iou_thr):
    """preds: (N,4) xyxy 已按 conf 降序；gts: (M,4)。
    返回长度 N 的 bool 数组（True=TP）。贪心：每个 GT 只被匹配一次。"""
    n = len(preds)
    flags = np.zeros(n, dtype=bool)
    if len(gts) == 0 or n == 0:
        return flags
    ious = iou_matrix(preds, gts)
    used = np.zeros(len(gts), dtype=bool)
    for i in range(n):
        row = ious[i].copy()
        row[used] = -1.0
        j = int(np.argmax(row))
        if row[j] >= iou_thr:
            flags[i] = True
            used[j] = True
    return flags


def average_precision(tp, n_gt):
    """101 点插值 AP。tp: bool 数组（已按 conf 降序）"""
    if n_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    tp_cum = np.cumsum(tp.astype(np.float64))
    fp_cum = np.cumsum((~tp).astype(np.float64))
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.linspace(0, 1, 101)
    return float(np.mean(np.interp(idx, mrec, mpre)))


# ---------------- COCO 读取 ----------------

def load_coco(coco_path, images_dir, want_cat="ship"):
    with open(coco_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    cats = {c["id"]: c["name"] for c in d["categories"]}
    keep = [cid for cid, nm in cats.items() if nm.lower() == want_cat.lower()]
    if not keep:
        raise SystemExit("标注里找不到类别 %r，实际类别：%s" % (want_cat, cats))
    by_img = defaultdict(list)
    for a in d["annotations"]:
        if a["category_id"] not in keep:
            continue
        if a.get("iscrowd", 0):
            continue
        x, y, w, h = a["bbox"]
        if w <= 0 or h <= 0:
            continue
        by_img[a["image_id"]].append([x, y, x + w, y + h])
    items = []
    for im in d["images"]:
        p = os.path.join(images_dir, im["file_name"])
        if not os.path.exists(p):
            continue
        gts = np.array(by_img.get(im["id"], []), dtype=np.float32).reshape(-1, 4)
        items.append({
            "path": p,
            "file_name": im["file_name"],
            "w": im["width"],
            "h": im["height"],
            "gt": gts,
        })
    return items


# ---------------- 主流程 ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="实验名，用于输出文件名")
    ap.add_argument("--images", required=True)
    ap.add_argument("--coco", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--conf", type=float, default=0.001, help="低阈值以画完整 PR 曲线")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS 的 IoU")
    ap.add_argument("--max-det", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0, help="只测前 N 张（0=全部）")
    ap.add_argument("--report-conf", type=float, default=0.30, help="报告 P/R 时用的置信度")
    ap.add_argument("--vis", type=int, default=6, help="保存几张可视化")
    ap.add_argument("--outdir", default=r"D:\rs-yolo\runs\crossdomain")
    args = ap.parse_args()

    from ultralytics import YOLO

    os.makedirs(args.outdir, exist_ok=True)
    items = load_coco(args.coco, args.images)
    if not items:
        raise SystemExit("没有可用样本（检查 --images 路径）")
    n_gt_total = sum(len(it["gt"]) for it in items)
    print("[数据] %d 张图，%d 个 GT 框（类别 ship）" % (len(items), n_gt_total))

    if args.limit:
        random.Random(0).shuffle(items)
        items = items[: args.limit]
        n_gt_total = sum(len(it["gt"]) for it in items)
        print("[数据] 抽样 %d 张，%d 个 GT 框" % (len(items), n_gt_total))

    model = YOLO(args.model)
    print("[模型] %s  imgsz=%d" % (args.model, args.imgsz))

    # 收集所有预测（全局按 conf 排序）+ 每张的 GT
    all_preds = []   # (conf, img_idx, box)
    per_img_pred = {}
    for i, it in enumerate(items):
        res = model.predict(
            source=it["path"], imgsz=args.imgsz, conf=args.conf, iou=args.iou,
            max_det=args.max_det, verbose=False, device=0,
        )[0]
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)
            conf = res.boxes.conf.cpu().numpy().astype(np.float32)
            order = np.argsort(-conf)
            xyxy, conf = xyxy[order], conf[order]
        else:
            xyxy = np.zeros((0, 4), dtype=np.float32)
            conf = np.zeros((0,), dtype=np.float32)
        per_img_pred[i] = (xyxy, conf)
        for k in range(len(conf)):
            all_preds.append((float(conf[k]), i, xyxy[k]))
        if (i + 1) % 50 == 0:
            print("  已推理 %d/%d" % (i + 1, len(items)))

    # ---- mAP50 / mAP50-95（单类别）----
    # 全局按 conf 降序排序后逐条判定 TP/FP，是 COCO 的标准做法
    all_preds.sort(key=lambda z: -z[0])
    ap_by_thr = {}
    stats = {}
    for thr in [round(0.5 + 0.05 * i, 2) for i in range(10)]:
        tp_flags = []
        used = {i: np.zeros(len(items[i]["gt"]), dtype=bool) for i in range(len(items))}
        ious_cache = {}
        for conf, idx, box in all_preds:
            gts = items[idx]["gt"]
            if len(gts) == 0:
                tp_flags.append(False)
                continue
            # 每条预测的框不同，必须逐条算 IoU（曾误用缓存，导致 mAP 算错）
            row = iou_matrix(box.reshape(1, 4), gts)[0]
            row[used[idx]] = -1.0
            j = int(np.argmax(row))
            if row[j] >= thr:
                tp_flags.append(True)
                used[idx][j] = True
            else:
                tp_flags.append(False)
        ap_by_thr[thr] = average_precision(np.array(tp_flags, dtype=bool), n_gt_total)

    map50 = ap_by_thr.get(0.5, 0.0)
    map5095 = float(np.nanmean([ap_by_thr[t] for t in sorted(ap_by_thr)]))

    # ---- 指定置信度下的 P / R ----
    tp = fp = 0
    for i, it in enumerate(items):
        xyxy, conf = per_img_pred[i]
        m = conf >= args.report_conf
        p_boxes = xyxy[m]
        gts = it["gt"]
        if len(p_boxes) == 0:
            continue
        flags = match_flags(p_boxes, gts, 0.5)
        tp += int(flags.sum())
        fp += int((~flags).sum())
    recall = tp / n_gt_total if n_gt_total else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")

    summary = {
        "name": args.name,
        "model": args.model,
        "imgsz": args.imgsz,
        "n_images": len(items),
        "n_gt": n_gt_total,
        "map50": round(map50, 4),
        "map50_95": round(map5095, 4),
        "report_conf": args.report_conf,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "tp": tp,
        "fp": fp,
        "fn": n_gt_total - tp,
        "ap_by_iou": {str(k): round(v, 4) for k, v in ap_by_thr.items()},
    }
    outp = os.path.join(args.outdir, args.name + ".json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print()
    print("========== %s ==========" % args.name)
    print("图片 %d 张 / GT %d 框" % (len(items), n_gt_total))
    print("mAP50    = %.4f" % map50)
    print("mAP50-95 = %.4f" % map5095)
    print("conf>=%.2f 时：P=%.4f R=%.4f (TP %d / FP %d / FN %d)"
          % (args.report_conf, precision, recall, tp, fp, n_gt_total - tp))
    print("结果已写入 %s" % outp)

    # ---- 可视化 ----
    if args.vis > 0:
        from PIL import Image, ImageDraw
        visdir = os.path.join(args.outdir, "vis_" + args.name)
        os.makedirs(visdir, exist_ok=True)
        # 挑 GT 数量适中的几张
        cand = sorted(range(len(items)), key=lambda i: -len(items[i]["gt"]))
        cand = [i for i in cand if len(items[i]["gt"]) > 0][: args.vis]
        for i in cand:
            it = items[i]
            im = Image.open(it["path"]).convert("RGB")
            dr = ImageDraw.Draw(im)
            for b in it["gt"]:
                dr.rectangle([b[0], b[1], b[2], b[3]], outline=(0, 255, 0), width=2)
            xyxy, conf = per_img_pred[i]
            for k in range(len(conf)):
                if conf[k] < args.report_conf:
                    continue
                b = xyxy[k]
                dr.rectangle([b[0], b[1], b[2], b[3]], outline=(255, 0, 0), width=2)
            dr.text((6, 6), "GT green / Pred red  GT=%d" % len(it["gt"]), fill=(255, 255, 0))
            im.save(os.path.join(visdir, "%03d_%s" % (i, os.path.basename(it["path"]))), quality=90)
        print("可视化 %d 张 -> %s" % (len(cand), visdir))

    return summary


if __name__ == "__main__":
    sys.exit(0 if main() else 0)
