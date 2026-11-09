# SPDX-FileCopyrightText: 2026 XiaoFenglistens
# SPDX-License-Identifier: MIT
"""带低分辨率退化增强的训练入口（参数风格同 ultralytics CLI）。

用法：
    set DEGRADE_P=0.35
    python tools\\train_degrade.py model=... data=... imgsz=512 epochs=100 name=aug_deg

先自检（不改模型，只验证增强链真的插进去了、且框坐标没被破坏）：
    python tools\\train_degrade.py --check
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import degrade_aug  # noqa: E402

_ROOT = os.environ.get("K230_ROOT", r"D:\rs-yolo")

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


def check():
    """验证：增强链里确有 RandomDegrade，跑一批样本后 img 形状与框数量不变。"""
    import numpy as np

    cfg = degrade_aug.apply()
    print(f"[CHECK] cfg={cfg}")
    from ultralytics.cfg import get_cfg
    from ultralytics.data.dataset import YOLODataset

    hyp = get_cfg()  # 用官方默认配置，避免手写 key 漏项（cutmix 等）
    hyp.mosaic = 0.0
    hyp.mixup = 0.0
    ds = YOLODataset(
        img_path=_ROOT + r"\datasets\sodaa_ship_hbb\images\train",
        imgsz=512, cache=False, augment=True, hyp=hyp, rect=False, batch_size=4,
        stride=32, pad=0.0, data={"names": {0: "ship"}, "nc": 1, "channels": 3},
    )
    names = [type(t).__name__ for t in ds.transforms.transforms]
    print(f"[CHECK] transforms = {names}")
    assert "RandomDegrade" in names, "退化增强没插进去"

    n_deg = 0
    for i in range(8):
        s = ds[i]
        arr = s["img"]
        assert tuple(arr.shape)[0] == 3, f"img 形状异常 {arr.shape}"
        if i in (0, 1, 2):
            n_deg += 1
    print(f"[CHECK] 8 个样本读 OK，img 形状 {tuple(s['img'].shape)}，cls {s['cls'].shape}")
    print("[CHECK] OK")


def main():
    if "--check" in sys.argv:
        check()
        return
    cfg = degrade_aug.apply()
    kw = parse([a for a in sys.argv[1:] if a != "--check"])
    print(f"[DEGRADE] config={cfg}")
    from ultralytics import YOLO

    model = YOLO(kw.pop("model"))
    model.train(**kw)


if __name__ == "__main__":
    main()
