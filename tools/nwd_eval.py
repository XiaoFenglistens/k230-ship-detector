# SPDX-FileCopyrightText: 2026 XiaoFenglistens
# SPDX-License-Identifier: MIT
"""NWD 消融实验的评测汇总：整体 mAP + 分尺寸 AP50。

用法：
    python tools\\nwd_eval.py nwd_a0 nwd_a50 nwd_a100 nwd_aadapt
输出：
    runs\\nwd_eval.json  与控制台 markdown 表
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ap_by_size  # noqa: E402
from ultralytics import YOLO  # noqa: E402

_ROOT = os.environ.get("K230_ROOT", r"D:\rs-yolo")

ROOT = _ROOT + r""
DATA = os.path.join(ROOT, "datasets", "sodaa_ship_hbb", "data.yaml")
IMGSZ = 512
OUT = os.path.join(ROOT, "runs", "nwd_eval.json")


def overall(weights):
    m = YOLO(weights)
    r = m.val(data=DATA, imgsz=IMGSZ, conf=0.001, iou=0.45, max_det=1000, split="val", verbose=False, plots=False)
    b = r.box
    return {
        "mAP50": float(b.map50),
        "mAP50_95": float(b.map),
        "precision": float(b.mp),
        "recall": float(b.mr),
    }


def main(tags):
    res = {}
    if os.path.exists(OUT):
        try:
            res = json.load(open(OUT, encoding="utf-8"))
        except Exception:
            res = {}
    for tag in tags:
        w = os.path.join(ROOT, "runs", tag, "weights", "best.pt")
        if not os.path.exists(w):
            print(f"[skip] {tag}: {w} 不存在")
            continue
        print(f"=== {tag} ===", flush=True)
        r = {"weights": w}
        r.update(overall(w))
        sz = ap_by_size.run(w, IMGSZ, tag)
        r["by_size"] = {k: v["ap50"] for k, v in sz["buckets"].items()}
        r["recall_by_size"] = {k: v["recall"] for k, v in sz["buckets"].items()}
        r["n_gt"] = {k: v["n_gt"] for k, v in sz["buckets"].items()}
        res[tag] = r
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)

    # 汇总表
    cols = ["mAP50", "mAP50_95", "precision", "recall"]
    bks = ("tiny", "small", "medium", "large")
    print("\n| run | " + " | ".join(cols) + " | " + " | ".join(f"AP {k}" for k in bks) + " | " + " | ".join(f"R {k}" for k in bks) + " |")
    print("|---" * (1 + len(cols) + 2 * len(bks)) + "|")
    for tag, r in res.items():
        bs = r.get("by_size", {})
        rc = r.get("recall_by_size", {})
        row = [f"{r.get(c, float('nan')):.4f}" for c in cols]
        row += ["-" if bs.get(k) is None else f"{bs[k]:.4f}" for k in bks]
        row += ["-" if rc.get(k) is None else f"{rc[k]:.4f}" for k in bks]
        print(f"| {tag} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main(sys.argv[1:] or ["nwd_a0", "nwd_a50", "nwd_a100", "nwd_aadapt"])
