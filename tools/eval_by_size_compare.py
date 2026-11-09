# SPDX-FileCopyrightText: 2026 XiaoFenglistens
# SPDX-License-Identifier: MIT
"""按目标尺寸分桶的检测评测 —— 用于"增益是否落在预期桶"的机制验证。

为什么需要它：
    NWD 的主张是"改善小目标匹配"。如果这个主张成立，增益就必须**集中在 8-16px 桶**，
    并且在 <8px 桶里几乎不出现（物理层死了，救不了）。只看总体 mAP 涨 1 点，
    无法区分"机制生效"和"随机波动"。

做法：
    每个尺寸桶当成一个独立"类别"，各自跑一遍贪心匹配（IoU>=0.5）——
    这样匹配到别的桶上的预测就会自然变成该桶的 FP，AP 不会被虚高。
    （踩过的坑：只统计"被匹配到的预测"，会让每桶 AP 都接近 0.99。）

输出：
    1) 终端表格：总体 mAP50 + 四桶 recall / AP50
    2) 两个及以上模型时打印逐桶增益 Δ 与机制检验（最大增益是否落在 8-16px）
    3) JSON 报告

用法：
    python tools\\eval_by_size_compare.py --models base=runs\\ship_512_v1\\weights\\best.pt --n 300
    python tools\\eval_by_size_compare.py --models base=...pt nwd=...pt --n 300 --imgsz 512
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.environ.get("K230_ROOT", r"D:\rs-yolo")
VAL_IMG = os.path.join(ROOT, "datasets", "sodaa_ship_hbb", "images", "val")
VAL_LBL = os.path.join(ROOT, "datasets", "sodaa_ship_hbb", "labels", "val")

BUCKETS = [("<8", 0.0, 8.0), ("8-16", 8.0, 16.0), ("16-32", 16.0, 32.0), (">32", 32.0, 1e9)]
EXPECT_BUCKET = "8-16"


def load_gt(label_path, w=800, h=800):
    boxes, sizes = [], []
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            for ln in f:
                p = ln.split()
                if len(p) < 5:
                    continue
                cx, cy, bw, bh = [float(v) for v in p[1:5]]
                W, H = bw * w, bh * h
                boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                              (cx + bw / 2) * w, (cy + bh / 2) * h])
                sizes.append(float(np.sqrt(max(1e-6, W * H))))
    except OSError:
        pass
    return np.array(boxes, np.float32).reshape(-1, 4), np.array(sizes, np.float32)


def bucket_index(sizes):
    idx = np.full(len(sizes), -1, np.int32)
    for i, (_n, lo, hi) in enumerate(BUCKETS):
        idx[(sizes >= lo) & (sizes < hi)] = i
    return idx


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = aa[:, None] + bb[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)


def ap_101(conf, is_tp, n_gt):
    """conf/is_tp 为全局预测列表；按 conf 降序算 101 点插值 AP。"""
    if n_gt == 0 or len(conf) == 0:
        return 0.0
    order = np.argsort(-np.asarray(conf, dtype=np.float64))
    tp = np.asarray(is_tp, dtype=bool)[order]
    ctp = np.cumsum(tp).astype(np.float64)
    cfp = np.cumsum(~tp).astype(np.float64)
    rec = ctp / float(n_gt)
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    grid = np.linspace(0, 1, 101)
    return float(np.mean(np.interp(grid, mrec, mpre)))


def greedy_match(pred_boxes, pred_conf, gt_boxes):
    """按置信度降序贪心匹配，返回每个预测的 is_tp 布尔数组与命中 GT 下标。"""
    n_p = len(pred_boxes)
    is_tp = np.zeros(n_p, bool)
    hit = np.full(n_p, -1, np.int32)
    if n_p == 0 or len(gt_boxes) == 0:
        return is_tp, hit
    order = np.argsort(-pred_conf)
    iou_full = iou_matrix(pred_boxes, gt_boxes)
    used = np.zeros(len(gt_boxes), bool)
    for k in order:
        row = iou_full[k].copy()
        row[used] = -1.0
        j = int(np.argmax(row))
        if row[j] >= 0.5:
            used[j] = True
            is_tp[k] = True
            hit[k] = j
    return is_tp, hit


def eval_one(name, model_path, imgsz, n_imgs, conf, iou_thr, max_det, seed):
    from ultralytics import YOLO
    t0 = time.time()
    model = YOLO(model_path)

    names = sorted(f for f in os.listdir(VAL_IMG) if f.lower().endswith(".jpg"))
    if n_imgs and n_imgs < len(names):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(names), n_imgs, replace=False)
        names = [names[i] for i in sorted(idx)]

    # 第一遍：把预测与 GT 全部缓存下来（只推理一次），后面每个桶复用
    cache = []
    preds = model.predict(
        source=[os.path.join(VAL_IMG, n) for n in names],
        imgsz=imgsz, conf=conf, iou=iou_thr, max_det=max_det, verbose=False, stream=True,
    )
    for fname, res in zip(names, preds):
        gt, sizes = load_gt(os.path.join(VAL_LBL, fname[:-4] + ".txt"))
        b = res.boxes
        if b is None or len(b) == 0:
            pb = np.zeros((0, 4), np.float32)
            pc = np.zeros((0,), np.float32)
        else:
            pb = b.xyxy.cpu().numpy().astype(np.float32)
            pc = b.conf.cpu().numpy().astype(np.float32)
        cache.append((pb, pc, gt, bucket_index(sizes)))

    out = {"name": name, "model": model_path, "imgsz": imgsz, "n_images": len(names)}

    # 总体
    all_conf, all_tp, n_gt_all, n_tp_all = [], [], 0, 0
    for pb, pc, gt, _bi in cache:
        is_tp, _ = greedy_match(pb, pc, gt)
        all_conf.extend(pc.tolist())
        all_tp.extend(is_tp.tolist())
        n_gt_all += len(gt)
        n_tp_all += int(is_tp.sum())
    out["overall"] = {"n_gt": n_gt_all,
                      "recall": n_tp_all / max(1, n_gt_all),
                      "ap50": ap_101(all_conf, all_tp, n_gt_all)}

    # 每个桶单独算 AP：按 COCO 的"忽略"语义 ——
    #   预测匹配到本桶 GT -> TP；匹配到别桶 GT -> 忽略；没匹配上 -> FP。
    # （若把"匹配到别桶"也当 FP，单一类别的检测器会让每桶 AP 都显得很低，没有可比性。）
    out["buckets"] = {}
    for bi, (bname, _lo, _hi) in enumerate(BUCKETS):
        confs, tps, n_gt, n_tp = [], [], 0, 0
        for pb, pc, gt, b_idx in cache:
            sel = b_idx == bi
            n_gt += int(sel.sum())
            if len(pb) == 0:
                continue
            is_tp, hit = greedy_match(pb, pc, gt)
            for k in range(len(pb)):
                if hit[k] >= 0:
                    if b_idx[hit[k]] == bi:
                        confs.append(float(pc[k]))
                        tps.append(True)
                        n_tp += 1
                    # 匹配到别桶 -> 忽略
                else:
                    confs.append(float(pc[k]))
                    tps.append(False)
        out["buckets"][bname] = {
            "n_gt": int(n_gt),
            "recall": (n_tp / n_gt) if n_gt else 0.0,
            "ap50": ap_101(confs, tps, n_gt),
        }

    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=path，可多个；第一个作为基线")
    ap.add_argument("--n", type=int, default=300, help="抽样图数（0=全量 698）")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--max-det", type=int, default=1000, help="必须 >= 单图最多目标数(668)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=os.path.join(ROOT, "runs", "eval_by_size.json"))
    args = ap.parse_args()

    specs = []
    for s in args.models:
        nm, pth = (s.split("=", 1) if "=" in s else (os.path.basename(s), s))
        if not os.path.exists(pth):
            raise SystemExit("找不到模型: %s" % pth)
        specs.append((nm, pth))

    results = []
    for nm, pth in specs:
        print("[跑] %s <- %s" % (nm, os.path.basename(pth)))
        results.append(eval_one(nm, pth, args.imgsz, args.n, args.conf, args.iou, args.max_det, args.seed))
        r = results[-1]
        print("     mAP50 %.4f | recall %.4f | %ds" % (r["overall"]["ap50"], r["overall"]["recall"], r["elapsed_s"]))

    print("\n=== 分桶结果（imgsz=%d, n=%d, conf=%.3f）===" % (args.imgsz, results[0]["n_images"], args.conf))
    print("%-14s %8s %8s | %-20s %-20s %-20s %-20s" % (
        "model", "mAP50", "recall", *["%s px" % b[0] for b in BUCKETS]))
    print("-" * 122)
    for r in results:
        cells = []
        for b in BUCKETS:
            v = r["buckets"][b[0]]
            cells.append("R%.3f/A%.3f (n%5d)" % (v["recall"], v["ap50"], v["n_gt"]))
        print("%-14s %8.4f %8.4f | %-20s %-20s %-20s %-20s" % (
            r["name"], r["overall"]["ap50"], r["overall"]["recall"], *cells))

    if len(results) >= 2:
        base = results[0]
        print("\n=== 逐桶增益（相对 %s）===" % base["name"])
        print("%-14s %11s |" % ("model", "Δ mAP50"), end="")
        for b in BUCKETS:
            print(" %14s" % ("Δrecall %s" % b[0]), end="")
        print()
        for r in results[1:]:
            print("%-14s %+11.4f |" % (r["name"], r["overall"]["ap50"] - base["overall"]["ap50"]), end="")
            for b in BUCKETS:
                d = r["buckets"][b[0]]["recall"] - base["buckets"][b[0]]["recall"]
                print(" %+14.4f" % d, end="")
            print()

        print("\n=== 机制检验（预期：最大增益落在 %s 桶）===" % EXPECT_BUCKET)
        for r in results[1:]:
            deltas = {b[0]: r["buckets"][b[0]]["recall"] - base["buckets"][b[0]]["recall"]
                      for b in BUCKETS}
            best_b = max(deltas, key=deltas.get)
            ok = "PASS" if best_b == EXPECT_BUCKET else "FAIL（最大增益在 %s）" % best_b
            print("  %-12s 最大增益桶=%s (%+.4f)  预期=%s  -> %s" % (r["name"], best_b, deltas[best_b], EXPECT_BUCKET, ok))
            print("               " + "  ".join("%s %+.4f" % (k, v) for k, v in deltas.items()))

    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "results": results}, f, ensure_ascii=False, indent=2)
    print("\nJSON -> %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
