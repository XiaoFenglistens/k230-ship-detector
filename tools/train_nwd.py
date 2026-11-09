# SPDX-FileCopyrightText: 2026 XiaoFenglistens
# SPDX-License-Identifier: MIT
"""带 NWD 补丁的训练入口。

用法（参数与 ultralytics CLI 同风格，key=value）：
    python tools\\train_nwd.py imgsz=512 epochs=100 batch=16 device=0 seed=0 \\
        model=D:\\rs-yolo\\weights\\yolov8n.pt data=D:\\rs-yolo\\datasets\\sodaa_ship_hbb\\data.yaml \\
        project=D:\\rs-yolo\\runs name=nwd_a50

NWD 行为由环境变量 NWD_ALPHA / NWD_MODE / NWD_C 控制（见 nwd_patch.py）。
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nwd_patch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

DEFAULTS = {
    "task": "detect",
    "mode": "train",
    "epochs": 100,
    "batch": 16,
    "workers": 4,
    "device": 0,
    "seed": 0,
    "plots": False,
    "exist_ok": True,
}


def parse(argv):
    kw = dict(DEFAULTS)
    for a in argv:
        if "=" not in a:
            continue
        k, v = a.split("=", 1)
        if k in ("imgsz", "epochs", "batch", "workers", "device", "seed"):
            kw[k] = int(v)
        elif k in ("plots", "exist_ok", "val"):
            kw[k] = v.lower() in ("1", "true", "yes")
        elif k in ("lr0", "lrf", "weight_decay", "box", "cls", "dfl"):
            kw[k] = float(v)
        else:
            kw[k] = v
    return kw


def main():
    cfg = nwd_patch.apply()
    kw = parse(sys.argv[1:])
    print(f"[NWD] config={cfg}")
    print(f"[NWD] train args={ {k: v for k, v in kw.items() if k in ('imgsz', 'epochs', 'batch', 'model', 'name', 'data')} }")
    t0 = time.time()
    model = YOLO(kw.pop("model"))
    results = model.train(**kw)
    print(f"[NWD] done in {(time.time() - t0) / 60:.1f} min")
    return results


if __name__ == "__main__":
    main()
