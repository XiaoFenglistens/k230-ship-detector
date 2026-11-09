# SPDX-FileCopyrightText: 2026 XiaoFenglistens
# SPDX-License-Identifier: MIT
"""20 类（DIOR）kmodel 与 onnx 的精度对拍：同图同参数算 mAP50 / mAP50-95。

与单类版 tools/quant_gap.py 的区别：
  * 输出是 (1, 4+20, N)，按类别分别做 NMS 与匹配，再对「GT 数＞0」的类别取均值；
  * GT 带类别号，只与同类预测匹配。

用法：
  python tools/quant_gap_mc.py --kmodel runs/dior20_416/weights/dior20_a16.kmodel \
      --out runs/quant_gap_dior_a16.json -n 50
"""
import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
_ROOT = os.environ.get("K230_ROOT", r"D:\rs-yolo")

ROOT = _ROOT + r""
DATA = os.path.join(ROOT, "datasets", "dior20")
IMG_DIR = os.path.join(DATA, "images", "val")
LBL_DIR = os.path.join(DATA, "labels", "val")
ONNX = os.path.join(ROOT, "runs", "dior20_416", "weights", "best.onnx")
SIZE = 416
NC = 20
CONF = 0.001
NMS_IOU = 0.45
MAX_DET = 1200

import nncase  # noqa: E402
import onnxruntime as ort  # noqa: E402



def letterbox(img, size):
    h, w = img.shape[:2]
    r = min(size / float(w), size / float(h))
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, np.uint8)
    padl, padt = (size - nw) // 2, (size - nh) // 2
    canvas[padt:padt + nh, padl:padl + nw] = resized
    return canvas, r, (padl, padt)


def nms(boxes, scores, iou_thres):
    if len(boxes) == 0:
        return np.zeros((0,), np.int64)
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
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][ovr <= iou_thres]
    return np.array(keep, np.int64)


def decode(out, conf=CONF, iou=NMS_IOU, max_det=MAX_DET):
    """out (1, 4+NC, N) -> list of (cls, score, xyxy)。按类别分别 NMS。"""
    p = out[0].transpose(1, 0)
    boxes_xywh = p[:, :4]
    cls_scores = p[:, 4:4 + NC]
    res = []
    for c in range(NC):
        s_all = cls_scores[:, c]
        m = s_all >= conf
        if not m.any():
            continue
        b, s = boxes_xywh[m], s_all[m]
        cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        k = nms(boxes, s, iou)[:max_det]
        for j in k:
            res.append((c, float(s[j]), boxes[j]))
    return res


def load_gt(txt_path, W, H):
    out = []
    if not os.path.exists(txt_path):
        return out
    with open(txt_path) as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue
            c = int(float(p[0]))
            cx, cy, w, h = [float(v) for v in p[1:5]]
            out.append((c, [(cx - w / 2) * W, (cy - h / 2) * H,
                            (cx + w / 2) * W, (cy + h / 2) * H]))
    return out


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


def ap_per_class(preds, gts, iou_thr):
    """preds: [(conf, img_idx, cls, box)]；gts: [[(cls, box), ...], ...]"""
    per_class = {}
    for c in range(NC):
        pc = sorted([p for p in preds if p[2] == c], key=lambda t: -t[0])
        n_gt_c = sum(1 for g in gts for (gc, _b) in g if gc == c)
        if n_gt_c == 0:
            continue
        used = {}
        for gi, g in enumerate(gts):
            idxs = [k for k, (gc, _b) in enumerate(g) if gc == c]
            used[gi] = np.zeros(len(g), bool)
            used[gi][[i for i in range(len(g)) if i not in idxs]] = True  # 非本类占位
        tp_flags = []
        for conf, gi, _c, box in pc:
            g = gts[gi]
            cand = [k for k, (gc, _b) in enumerate(g) if gc == c and not used[gi][k]]
            if not cand:
                tp_flags.append(False)
                continue
            gb = np.array([g[k][1] for k in cand], np.float32)
            ious = iou_matrix(box.reshape(1, 4), gb)[0]
            best = int(np.argmax(ious))
            if ious[best] >= iou_thr:
                used[gi][cand[best]] = True
                tp_flags.append(True)
            else:
                tp_flags.append(False)
        per_class[c] = average_precision(tp_flags, n_gt_c)
    vals = [v for v in per_class.values() if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan"), per_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kmodel", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("-n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    rng = np.random.RandomState(args.seed)
    if len(files) > args.n:
        idxs = np.sort(rng.choice(len(files), args.n, replace=False))
        files = [files[i] for i in idxs]
    print("[数据] %d 张 DIOR val 图" % len(files), flush=True)

    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    sim = nncase.Simulator()
    with open(args.kmodel, "rb") as f:
        sim.load_model(f.read())
    print("[模型] %s" % os.path.basename(args.kmodel), flush=True)

    gts, preds_onnx, preds_km = [], [], []
    t0 = time.time()
    for i, path in enumerate(files):
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        H, W = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        lb, r, (padl, padt) = letterbox(rgb, SIZE)
        gts.append(load_gt(os.path.join(LBL_DIR, os.path.basename(path)[:-4] + ".txt"), W, H))

        x_u8 = np.transpose(lb, (2, 0, 1))[None].astype(np.uint8)
        x_f32 = x_u8.astype(np.float32) / 255.0

        o_onnx = sess.run(None, {in_name: x_f32})[0]
        sim.set_input_tensor(0, nncase.RuntimeTensor.from_numpy(x_u8))
        sim.run()
        o_km = sim.get_output_tensor(0).to_numpy()

        for out, bag in ((o_onnx, preds_onnx), (o_km, preds_km)):
            for c, s, b in decode(out):
                b = b.copy()
                b[[0, 2]] = (b[[0, 2]] - padl) / r
                b[[1, 3]] = (b[[1, 3]] - padt) / r
                bag.append((s, i, c, b))

        if (i + 1) % 5 == 0 or i == len(files) - 1:
            el = time.time() - t0
            print("  %d/%d  %.0fs  (%.1fs/张)  pred onnx=%d km=%d"
                  % (i + 1, len(files), el, el / (i + 1), len(preds_onnx), len(preds_km)),
                  flush=True)
            try:
                with open(args.out + ".progress", "w", encoding="utf-8") as pf:
                    pf.write("%d/%d elapsed=%.0fs %.1fs/img\n"
                             % (i + 1, len(files), el, el / (i + 1)))
            except Exception:
                pass

    thrs = [round(0.5 + 0.05 * k, 2) for k in range(10)]
    res = {}
    for name, bag in (("onnx", preds_onnx), ("kmodel", preds_km)):
        ap50, per50 = ap_per_class(bag, gts, 0.5)
        aps = [ap_per_class(bag, gts, t)[0] for t in thrs]
        res[name] = {
            "n_images": len(files),
            "n_pred": len(bag),
            "mAP50": ap50,
            "mAP50_95": float(np.mean(aps)),
            "ap50_per_class": {str(k): v for k, v in per50.items()},
        }
        print("\n===== %s =====" % name)
        print("  mAP50    = %.4f" % ap50)
        print("  mAP50-95 = %.4f" % float(np.mean(aps)))

    print("\n===== 量化损失（kmodel - onnx）=====")
    print("  d_mAP50    = %+.4f" % (res["kmodel"]["mAP50"] - res["onnx"]["mAP50"]))
    print("  d_mAP50-95 = %+.4f" % (res["kmodel"]["mAP50_95"] - res["onnx"]["mAP50_95"]))

    res["kmodel_path"] = args.kmodel
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("\n结果已写入 %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
