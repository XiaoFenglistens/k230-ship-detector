# SPDX-FileCopyrightText: 2026 XiaoFenglistens
# SPDX-License-Identifier: MIT
"""低分辨率退化数据增强（训练侧，零推理成本）。

动机：模型最终看到的是 800×480 摄像头画面缩到 512 之后的样子（0.64×），
而且中间还经过显示屏、镜头、JPEG 三重模糊。训练集却是干净的 800×800 航拍切片，
两者存在"清晰度域差"。这里在训练时随机把图弄糊，让模型对渣画质免疫。

几何不变：所有退化都是等比缩放回去 + 模糊 + JPEG 重压缩 + 轻亮度抖动，
**框坐标不受影响**，因此不需要改标签。

用法：
    set DEGRADE_P=0.35 & python tools\\train_degrade.py ...
环境变量：
    DEGRADE_P      每张图施加退化的概率（默认 0.35）
    DEGRADE_DOWN   降采样系数下限（默认 0.35，即最糊时先缩到 35% 再放大）
"""

import os
import random

import cv2
import numpy as np

from ultralytics.data.augment import v8_transforms

DEGRADE_P = float(os.environ.get("DEGRADE_P", "0.35") or 0.35)
DOWN_MIN = float(os.environ.get("DEGRADE_DOWN", "0.35") or 0.35)


class RandomDegrade:
    """模拟低分辨率成像链路：降采样→回放→模糊→JPEG 重压缩→亮度抖动。"""

    def __init__(self, p=DEGRADE_P, down_min=DOWN_MIN):
        self.p = p
        self.down_min = down_min
        self.applied = 0
        self.total = 0

    def __call__(self, labels):
        self.total += 1
        if random.random() >= self.p:
            return labels
        img = labels.get("img")
        if img is None or img.ndim != 3:
            return labels
        h, w = img.shape[:2]
        out = img

        # 1) 降采样再放大（模拟传感器分辨率不足）
        s = random.uniform(self.down_min, 0.85)
        small = cv2.resize(out, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

        # 2) 轻微失焦
        if random.random() < 0.7:
            sigma = random.uniform(0.3, 1.2)
            k = max(3, int(round(sigma * 3)) | 1)
            out = cv2.GaussianBlur(out, (k, k), sigma)

        # 3) JPEG 重压缩（模拟相机/显示屏编码损失）
        if random.random() < 0.5:
            q = random.randint(40, 80)
            ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            if ok:
                out = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        # 4) 亮度/对比度抖动（模拟屏幕反光与曝光）
        if random.random() < 0.5:
            alpha = random.uniform(0.9, 1.1)
            beta = random.uniform(-12, 12)
            out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        labels["img"] = np.ascontiguousarray(out)
        self.applied += 1
        return labels


_orig_v8 = v8_transforms


def v8_transforms_degrade(dataset, imgsz, hyp, *args, **kwargs):
    """在标准 v8 增强链里插入退化增强（放在几何变换之前，最贴近真实成像顺序）。"""
    t = _orig_v8(dataset, imgsz, hyp, *args, **kwargs)
    try:
        new_list = list(t.transforms)
        # 插在 RandomHSV 之后（如果存在），否则插到最前
        idx = 0
        for i, tr in enumerate(new_list):
            if type(tr).__name__ == "RandomHSV":
                idx = i + 1
        new_list.insert(idx, RandomDegrade())
        t.transforms = new_list
    except Exception as e:  # 结构变了也不能让训练挂掉
        print(f"[DEGRADE] 注入失败，跳过：{e}")
    return t


_patched = False


def apply():
    global _patched
    import ultralytics.data.augment as A
    import ultralytics.data.dataset as D  # 它在模块顶部 from ... import v8_transforms，得单独换

    if _patched:
        return {"p": DEGRADE_P, "down_min": DOWN_MIN}
    A.v8_transforms = v8_transforms_degrade
    D.v8_transforms = v8_transforms_degrade
    _patched = True
    cfg = {"p": DEGRADE_P, "down_min": DOWN_MIN}
    print(f"[DEGRADE] patch applied: {cfg}")
    return cfg
