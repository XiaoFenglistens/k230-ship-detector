# SPDX-FileCopyrightText: 2026 XiaoFenglistens
# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""
速度分解图：把一帧的耗时按「取流 / 预处理(ai2d) / KPU推理 / 后处理 / 画框 / 显示」堆叠起来。

数据来源：D:\\rs-yolo\\runs\\profile_breakdown.json（板端 prof*.py 实测，每档 60 帧取均值）

用法：
  python tools\\profile_plot.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

RUNS = r"D:\rs-yolo\runs"
SRC = os.path.join(RUNS, "profile_breakdown.json")
OUT = os.path.join(RUNS, "profile_breakdown.png")

for name in ("Microsoft YaHei", "SimHei", "DengXian"):
    try:
        path = font_manager.findfont(font_manager.FontProperties(family=name),
                                     fallback_to_default=False)
        font_manager.fontManager.addfont(path)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

STAGES = [
    ("snapshot", "取流", "#a0aec0"),
    ("preprocess", "预处理 ai2d", "#63b3ed"),
    ("infer", "KPU 推理", "#2b6cb0"),
    ("postprocess", "后处理", "#68d391"),
    ("draw", "画框+标签", "#f6ad55"),
    ("show", "显示", "#b794f4"),
]


def main():
    with open(SRC, "r", encoding="utf-8") as f:
        data = json.load(f)

    runs = data["runs"]
    labels = [r["label"] for r in runs]
    fig, ax = plt.subplots(figsize=(12.6, 4.6))
    left = [0.0] * len(runs)
    for key, name, color in STAGES:
        vals = [r[key] for r in runs]
        ax.barh(labels, vals, left=left, color=color, label=name, height=0.55)
        for i, v in enumerate(vals):
            if v >= 1.5:
                ax.text(left[i] + v / 2, i, "%.1f" % v, va="center", ha="center",
                        fontsize=9, color="white", fontweight="bold")
        left = [left[i] + vals[i] for i in range(len(runs))]

    for i, r in enumerate(runs):
        ax.text(left[i] + 0.8, i, "合计 %.1f ms / %.1f fps" % (r["total"], r["fps"]),
                va="center", fontsize=10, color="#1a202c")

    ax.set_xlabel("单帧耗时 (ms)", fontsize=12)
    ax.set_title(data.get("title", "单帧耗时分解（K230 板端实测，每档 60 帧均值）"), fontsize=13)
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False, fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()
    ax.set_xlim(0, max(left) * 1.32)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print("图 -> %s" % OUT)


if __name__ == "__main__":
    main()
