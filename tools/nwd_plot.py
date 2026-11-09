# SPDX-FileCopyrightText: 2026 XiaoFenglistens
# SPDX-License-Identifier: MIT
"""NWD 消融实验汇总图：整体精度 + 分尺寸 AP50 + 分尺寸召回。

输入 runs\\nwd_eval.json（由 tools\\nwd_eval.py 生成），输出 runs\\nwd_ablation.png。
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

_ROOT = os.environ.get("K230_ROOT", r"D:\rs-yolo")

RUNS = _ROOT + r"\runs"
SRC = os.path.join(RUNS, "nwd_eval.json")
OUT = os.path.join(RUNS, "nwd_ablation.png")

for name in ("Microsoft YaHei", "SimHei", "DengXian"):
    try:
        path = font_manager.findfont(font_manager.FontProperties(family=name), fallback_to_default=False)
        font_manager.fontManager.addfont(path)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

LABELS = {
    "nwd_a0": "α=0\n(IoU 基线)",
    "nwd_a50": "α=0.5",
    "nwd_a100": "α=1.0\n(纯 NWD)",
    "nwd_aadapt": "尺寸自适应",
    "nwd_rank50": "rank 0.5\n(仅排序用)",
    "nwd_gate20": "gate 20px\n(仅小目标)",
}
ORDER = ["nwd_a0", "nwd_a50", "nwd_a100", "nwd_aadapt", "nwd_rank50", "nwd_gate20"]
COLORS = ["#94a3b8", "#60a5fa", "#2563eb", "#7c3aed", "#f59e0b", "#ef4444"]
BUCKETS = [("tiny", "<8px"), ("small", "8–16px"), ("medium", "16–32px"), ("large", ">32px")]


def main():
    res = json.load(open(SRC, encoding="utf-8"))
    tags = [t for t in ORDER if t in res]
    if not tags:
        raise SystemExit("nwd_eval.json 里没有可用结果")

    fig, axes = plt.subplots(1, 3, figsize=(20.5, 5.4))

    # ① 整体 mAP50 / mAP50-95
    ax = axes[0]
    x = np.arange(len(tags))
    w = 0.36
    v50 = [res[t]["mAP50"] for t in tags]
    v95 = [res[t]["mAP50_95"] for t in tags]
    b1 = ax.bar(x - w / 2, v50, w, label="mAP50", color="#2563eb")
    b2 = ax.bar(x + w / 2, v95, w, label="mAP50-95", color="#93c5fd")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.006, f"{r.get_height():.4f}",
                    ha="center", va="bottom", fontsize=8.5)
    base = res[tags[0]]["mAP50"]
    ax.axhline(base, color="#ef4444", ls="--", lw=1)
    ax.text(len(tags) - 0.5, base + 0.004, f"基线 {base:.4f}", color="#ef4444", fontsize=9, ha="right")
    ax.set_xticks(x, [LABELS.get(t, t) for t in tags], fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("精度", fontsize=11)
    ax.set_title("整体精度", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    # ② 分尺寸 AP50
    ax = axes[1]
    n = len(BUCKETS)
    w2 = 0.8 / max(len(tags), 1)
    for i, t in enumerate(tags):
        vals = [res[t]["by_size"].get(k) or 0 for k, _ in BUCKETS]
        ax.bar(np.arange(n) + (i - (len(tags) - 1) / 2) * w2, vals, w2,
               label=LABELS.get(t, t).replace("\n", " "), color=COLORS[i % len(COLORS)])
    ax.set_xticks(np.arange(n), [lab for _, lab in BUCKETS])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("AP50", fontsize=11)
    ax.set_title("分尺寸 AP50", fontsize=13)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    # ③ 分尺寸召回（conf=0.30）
    ax = axes[2]
    for i, t in enumerate(tags):
        vals = [res[t].get("recall_by_size", {}).get(k) or 0 for k, _ in BUCKETS]
        ax.bar(np.arange(n) + (i - (len(tags) - 1) / 2) * w2, vals, w2,
               label=LABELS.get(t, t).replace("\n", " "), color=COLORS[i % len(COLORS)])
    if len(tags) >= 2:
        r0 = res[tags[0]].get("recall_by_size", {}).get("small")
        rN = res[tags[-1]].get("recall_by_size", {}).get("small")
        if r0 and rN:
            ax.text(1, max(r0, rN) + 0.04, f"8–16px: {r0:.3f} → {rN:.3f}", ha="center",
                    fontsize=9, color="#b91c1c")
    ax.set_xticks(np.arange(n), [lab for _, lab in BUCKETS])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("召回率", fontsize=11)
    ax.set_title("分尺寸召回（conf=0.30）", fontsize=13)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("NWD 小目标标签分配消融（SODA-A ship，YOLOv8n @512，100 epoch，seed=0）", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT, dpi=125)
    print("saved", OUT, os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    main()
