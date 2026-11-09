# SPDX-FileCopyrightText: 2026 XiaoFenglistens
# SPDX-License-Identifier: MIT
"""画「为什么小目标的 IoU 会崩」——IoU 与 NWD 随像素偏移的变化对比。

不碰板子、不训练，纯数学。用来：
  1) 论证 NWD 的必要性（论文第二三节的图）
  2) 直观看出"同样的 IoU，小船只允许偏几像素"
  3) 定 C 的取值（C = 数据集目标平均绝对尺寸）

输出：D:\rs-yolo\runs\iou_vs_nwd.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = os.environ.get("K230_ROOT", r"D:\rs-yolo")
OUT = os.path.join(ROOT, "runs", "iou_vs_nwd.png")

# 中文字体（matplotlib 默认不含中文）
try:
    fp = font_manager.FontProperties(family="Microsoft YaHei")
    name = font_manager.findfont(fp, fallback_to_default=False)
    font_manager.fontManager.addfont(name)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=name).get_name()
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass


def iou_shift(w, h, dx, dy=0.0):
    """同尺寸框平移 (dx,dy) 后的 IoU。"""
    ix = max(0.0, w - abs(dx))
    iy = max(0.0, h - abs(dy))
    inter = ix * iy
    union = w * h * 2 - inter
    return inter / union if union > 0 else 0.0


def nwd_shift(w, h, dx, dy=0.0, C=22.0):
    """同尺寸框平移后的 NWD（尺寸项为 0，只剩中心距离项）。"""
    W2 = dx * dx + dy * dy
    return float(np.exp(-np.sqrt(W2) / C))


def main():
    C = 22.0
    shifts = np.arange(0, 25.01, 0.25)

    # 三档目标：tiny / 中位 / 大
    cases = [
        ("12x12 px（<8px 缩放后，基本无解）", 12, 12, "#d62728"),
        ("20x24 px（中位目标）", 20, 24, "#1f77b4"),
        ("60x60 px（较大目标）", 60, 60, "#2ca02c"),
        ("200x240 px（大目标）", 200, 240, "#9467bd"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))

    # 左：IoU 随偏移
    ax = axes[0]
    for label, w, h, c in cases:
        ys = [iou_shift(w, h, d) for d in shifts]
        ax.plot(shifts, ys, label=label, color=c, lw=2)
    ax.axhline(0.5, ls="--", c="gray", lw=1)
    ax.text(24.2, 0.52, "常用匹配阈值 0.5", ha="right", fontsize=9, color="gray")
    ax.axvline(8, ls=":", c="red", lw=1.2)
    ax.text(8.3, 0.05, "stride=8：特征图一格", fontsize=9, color="red")
    ax.set_xlabel("预测框相对真值框的像素偏移 (px)")
    ax.set_ylabel("IoU")
    ax.set_title("① IoU：同样的偏移，小目标崩得更快")
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")

    # 右：NWD 随偏移
    ax = axes[1]
    for label, w, h, c in cases:
        ys = [nwd_shift(w, h, d, C=C) for d in shifts]
        ax.plot(shifts, ys, label=label, color=c, lw=2)
    ax.axhline(0.0, ls="--", c="gray", lw=1)
    ax.axvline(8, ls=":", c="red", lw=1.2)
    ax.text(C, 0.02, "C = 22 px", fontsize=9, color="gray")
    ax.set_xlabel("预测框相对真值框的像素偏移 (px)")
    ax.set_ylabel("NWD（越大越相似）")
    ax.set_title("② NWD：与目标尺寸无关，全程有梯度")
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")

    fig.suptitle("为什么小目标会「匹配崩」：IoU 的对尺度敏感 vs NWD 的距离度量  "
                 "（C = 数据集目标平均尺寸 %.0f px）" % C, fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print("图 -> %s" % OUT)

    # 文字表格：同样的 IoU 需要多大偏移
    print("\n【关键对照】达到相同 IoU 所需的像素偏移：")
    for thr in (0.5, 0.7, 0.818):
        row = []
        for label, w, h, _ in cases:
            d = None
            for s in shifts:
                if iou_shift(w, h, s) < thr:
                    d = s
                    break
            row.append("%s -> %s px" % (label.split("（")[0], ("%.2f" % d) if d is not None else ">25"))
        print("  IoU=%.3f :  %s" % (thr, " | ".join(row)))

    print("\n【零重叠对比】两框完全不重叠、中心距 22 px 时：")
    for label, w, h, _ in cases:
        print("  %-32s IoU=%.3f  NWD=%.3f" % (label, iou_shift(w, h, max(w, h) + 1), nwd_shift(w, h, C * 1.0, C=C)))


if __name__ == "__main__":
    main()
