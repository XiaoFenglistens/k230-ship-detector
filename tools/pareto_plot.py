# -*- coding: utf-8 -*-
"""
精度-速度帕累托曲线：把 320 / 416 / 512 / 640 四档的「板上实测帧率」与「验证集 mAP」画在一张图上。

数据来源：
  1) 精度：D:\\rs-yolo\\runs\\ship_<size>_v1\\results.csv，取 mAP50-95 最高的那一轮
  2) 速度：D:\\rs-yolo\\runs\\pareto_board.json（板端 bench 脚本实测写入）

用法：
  python tools\\pareto_plot.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

RUNS = r"D:\rs-yolo\runs"
BOARD = os.path.join(RUNS, "pareto_board.json")
OUT = os.path.join(RUNS, "pareto_curve.png")
SIZES = [320, 416, 512, 640]

# ---- 中文字体
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


def best_from_csv(path):
    """返回 (mAP50, mAP50-95, epoch)，按 mAP50-95 取最优"""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        head = f.readline().strip().split(",")
        idx = {k.strip(): i for i, k in enumerate(head)}
        i_map = idx["metrics/mAP50(B)"]
        i_map95 = idx["metrics/mAP50-95(B)"]
        i_ep = idx["epoch"]
        best = None
        for line in f:
            p = line.strip().split(",")
            if len(p) <= i_map95:
                continue
            try:
                m50, m95, ep = float(p[i_map]), float(p[i_map95]), int(float(p[i_ep])) + 1
            except ValueError:
                continue
            if best is None or m95 > best[1]:
                best = (m50, m95, ep)
    return best


def main():
    with open(BOARD, "r", encoding="utf-8") as f:
        board = json.load(f)

    rows = []
    for s in SIZES:
        acc = best_from_csv(os.path.join(RUNS, "ship_%d_v1" % s, "results.csv"))
        spd = board.get(str(s))
        if acc is None or spd is None:
            print("[skip] imgsz=%d  精度=%s  速度=%s" % (s, acc is not None, spd is not None))
            continue
        rows.append({
            "imgsz": s,
            "mAP50": acc[0],
            "mAP50_95": acc[1],
            "epoch": acc[2],
            "fps": spd["fps"],
            "infer_ms": spd["infer_ms"],
            "kmodel_mb": spd.get("kmodel_mb"),
        })

    if not rows:
        raise SystemExit("没有可用数据")

    print("\n%-8s %-9s %-11s %-8s %-11s" % ("imgsz", "mAP50", "mAP50-95", "fps", "infer_ms"))
    for r in rows:
        print("%-8d %-9.4f %-11.4f %-8.1f %-11.1f" % (
            r["imgsz"], r["mAP50"], r["mAP50_95"], r["fps"], r["infer_ms"]))

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.6))
    for ax, key, title in (
        (axes[0], "mAP50", "mAP@0.5 vs 板端帧率"),
        (axes[1], "mAP50_95", "mAP@0.5:0.95 vs 板端帧率"),
    ):
        xs = [r["fps"] for r in rows]
        ys = [r[key] for r in rows]
        ax.plot(xs, ys, "-o", color="#2b6cb0", linewidth=2, markersize=9, zorder=3)
        for r in rows:
            ax.annotate("%d" % r["imgsz"], (r["fps"], r[key]),
                        textcoords="offset points", xytext=(0, 11),
                        ha="center", fontsize=11, fontweight="bold", color="#1a365d")
        # 标出 416（部署默认）
        for r in rows:
            if r["imgsz"] == 416:
                ax.plot([r["fps"]], [r[key]], "o", markersize=15,
                        markerfacecolor="none", markeredgecolor="#e53e3e",
                        markeredgewidth=2.2, zorder=4)
                ax.annotate("部署默认", (r["fps"], r[key]),
                            textcoords="offset points", xytext=(0, -22),
                            ha="center", fontsize=9, color="#e53e3e")
        ax.set_xlabel("板上实时帧率 (fps)", fontsize=12)
        ax.set_ylabel(title.split(" vs ")[0], fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(min(xs) - 2.0, max(xs) + 2.0)

    fig.suptitle("精度-速度帕累托曲线（SODA-A ship，K230 板端实测）", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT, dpi=130)
    print("\n图 -> %s" % OUT)

    with open(os.path.join(RUNS, "pareto_table.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("表 -> %s" % os.path.join(RUNS, "pareto_table.json"))


if __name__ == "__main__":
    main()
