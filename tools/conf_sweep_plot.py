# SPDX-FileCopyrightText: 2026 XiaoFenglistens
# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""
置信度阈值消融曲线图。

读 runs\\conf_sweep_416.json / conf_sweep_512.json，画：
  左：精度 / 召回 / F1 随阈值变化（标出部署用的 0.30 与最佳 F1 点）
  右：每图平均框数随阈值变化（这个数关系到板端 max_boxes_num）

用法：python tools\\conf_sweep_plot.py
"""
import os as _os
KROOT = _os.environ.get("K230_ROOT", ".")  # 项目根目录，默认当前目录
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

RUNS = _os.path.join(KROOT, 'runs')
OUT = os.path.join(RUNS, "conf_sweep.png")
DEPLOYED = 0.30

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


def load(size):
    p = os.path.join(RUNS, "conf_sweep_%d.json" % size)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    data = {s: load(s) for s in (416, 512) if os.path.exists(os.path.join(RUNS, "conf_sweep_%d.json" % s))}
    if not data:
        raise SystemExit("没有找到 conf_sweep_*.json")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))

    ax = axes[0]
    styles = {416: "-", 512: "--"}
    colors = {"precision": "#2b6cb0", "recall": "#dd6b20", "f1": "#2f855a"}
    names = {"precision": "精确率 P", "recall": "召回率 R", "f1": "F1"}
    for size, d in sorted(data.items()):
        xs = [r["conf"] for r in d["sweep"]]
        for key in ("precision", "recall", "f1"):
            ax.plot(xs, [r[key] for r in d["sweep"]], styles[size], color=colors[key],
                    marker="o", ms=3.5, lw=1.8,
                    label="%s · %d 输入" % (names[key], size))
        best = d["best_f1"]
        ax.plot([best["conf"]], [best["f1"]], "k*", ms=13, zorder=5)
        dy = 8 if size == 512 else -16
        ax.annotate("最佳 %d：%.2f" % (size, best["conf"]),
                    xy=(best["conf"], best["f1"]), xytext=(8, dy),
                    textcoords="offset points", fontsize=9, color="#1a202c")

    ax.axvline(DEPLOYED, color="crimson", ls=":", lw=2)
    ax.text(DEPLOYED + 0.008, 0.42, "部署取值 0.30", color="crimson", fontsize=10, rotation=90, va="bottom")
    ax.set_xlabel("置信度阈值", fontsize=12)
    ax.set_ylabel("指标值", fontsize=12)
    ax.set_title("精度 / 召回 / F1 随置信度阈值的变化（全量 val 698 图）", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.2, 1.0)
    ax.legend(fontsize=8, ncol=2, loc="lower left")

    ax2 = axes[1]
    for size, d in sorted(data.items()):
        xs = [r["conf"] for r in d["sweep"]]
        ax2.plot(xs, [r["boxes_per_image"] for r in d["sweep"]], styles[size],
                 color="#4a5568", marker="o", ms=4, lw=1.8, label="%d 输入" % size)
    ax2.axvline(DEPLOYED, color="crimson", ls=":", lw=2)
    ax2.set_xlabel("置信度阈值", fontsize=12)
    ax2.set_ylabel("平均每图输出框数", fontsize=12)
    ax2.set_title("每图框数随阈值的变化（关系到板端 max_boxes_num）", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print("图 -> %s" % OUT)


if __name__ == "__main__":
    main()
