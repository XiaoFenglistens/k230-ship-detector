# SPDX-FileCopyrightText: 2026 XiaoFenglistens
# SPDX-License-Identifier: MIT
"""生成"低分辨率基准集"用的投屏场景图。

做法：从 SODA-A val 里挑 3 张（密集/中等/稀疏各一），放大到 1440x1440，
居中贴到 2560x1440 的黑底画布上 —— 和之前正交实验用的 orth_screen.png 同一规格，
这样摄像头取景角度、屏幕占比都可复现。

输出：
  D:\rs-yolo\bench_scenes\scene1_2560x1440.png ... scene3_2560x1440.png
  D:\rs-yolo\bench_scenes\scenes.json   (场景 -> 原图/标签 映射 + 6 组拍摄计划)

用法：
  python tools\make_bench_scenes.py            # 默认 seed=0
  python tools\make_bench_scenes.py --seed 7
"""

import argparse
import json
import os
import random

import numpy as np
from PIL import Image

ROOT = os.environ.get("K230_ROOT", r"D:\rs-yolo")
VAL_IMG = os.path.join(ROOT, "datasets", "sodaa_ship_hbb", "images", "val")
VAL_LBL = os.path.join(ROOT, "datasets", "sodaa_ship_hbb", "labels", "val")
OUTDIR = os.path.join(ROOT, "bench_scenes")

SCREEN_W, SCREEN_H = 2560, 1440
CROP = 1440  # 正方形投放区边长


def count_boxes(lbl_path):
    try:
        with open(lbl_path, "r", encoding="utf-8") as f:
            return sum(1 for ln in f if ln.strip())
    except OSError:
        return -1


def pick(pool, want, seed):
    '''从候选里随机取一个（同密度档里随机，避免总挑同一个）。'''
    if not pool:
        return None
    rnd = random.Random(seed)
    return rnd.choice(sorted(pool))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not os.path.isdir(VAL_IMG):
        raise SystemExit("找不到 val 图目录: %s" % VAL_IMG)

    dense, medium, sparse = [], [], []
    for name in sorted(os.listdir(VAL_IMG)):
        if not name.lower().endswith(".jpg"):
            continue
        lbl = os.path.join(VAL_LBL, name[:-4] + ".txt")
        n = count_boxes(lbl)
        if n <= 0:
            continue
        if n >= 100:
            dense.append((name, n))
        elif n >= 20:
            medium.append((name, n))
        elif 3 <= n <= 15:
            # 稀疏档限定 3~15 框：只 1~2 框的图统计意义太弱，做基准集不合适
            sparse.append((name, n))

    print("候选：密集 %d / 中等 %d / 稀疏 %d" % (len(dense), len(medium), len(sparse)))

    chosen = [
        ("scene1", pick(dense, 1, args.seed + 1), "密集"),
        ("scene2", pick(medium, 1, args.seed + 2), "中等"),
        ("scene3", pick(sparse, 1, args.seed + 3), "稀疏"),
    ]

    os.makedirs(OUTDIR, exist_ok=True)
    canvas_bg = Image.new("RGB", (SCREEN_W, SCREEN_H), (0, 0, 0))
    scenes = []

    for key, item, label in chosen:
        if item is None:
            print("[跳过] %s 该密度档没有候选" % label)
            continue
        name, n = item
        src = os.path.join(VAL_IMG, name)
        lbl = os.path.join(VAL_LBL, name[:-4] + ".txt")
        im = Image.open(src).convert("RGB")
        # 800x800 -> 1440x1440（Lanczos 放大，模拟"投到大屏上"）
        big = im.resize((CROP, CROP), Image.LANCZOS)
        canvas = canvas_bg.copy()
        left = (SCREEN_W - CROP) // 2
        top = (SCREEN_H - CROP) // 2
        canvas.paste(big, (left, top))
        out_png = os.path.join(OUTDIR, "%s_%dx%d.png" % (key, SCREEN_W, SCREEN_H))
        canvas.save(out_png)
        scenes.append({
            "scene": key,
            "density": label,
            "boxes": n,
            "src_image": src,
            "src_label": lbl,
            "screen_png": out_png,
            "screen_rect": [left, top, CROP, CROP],
        })
        print("%s <- %s (%d 框, %s) -> %s" % (key, name, n, label, os.path.basename(out_png)))

    # 6 组拍摄计划：3 场景 x 2 距离
    plan = []
    g = 1
    for sc in scenes:
        for dist in ("near", "far"):
            plan.append({"group": g, "scene": sc["scene"], "dist": dist})
            g += 1

    meta = {
        "screen": [SCREEN_W, SCREEN_H],
        "crop": CROP,
        "scenes": scenes,
        "groups": plan,
        "note": "near 约 20-25cm，far 约 40-45cm；正对屏幕，板子下沿与屏幕下沿大致平行",
    }
    with open(os.path.join(OUTDIR, "scenes.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("\n拍摄计划：")
    for p in plan:
        print("  g%02d -> %s / %s" % (p["group"], p["scene"], p["dist"]))
    print("\n场景图与 scenes.json 已写入 %s" % OUTDIR)


if __name__ == "__main__":
    main()
