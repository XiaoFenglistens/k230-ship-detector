# -*- coding: utf-8 -*-
"""
汇总 runs\crossdomain\*.json 成一张跨域对比表 + 柱状图 + 可直接贴进文档的 Markdown 表。

用法：
  python tools\\crossdomain_summary.py
"""
import glob
import json
import os

OUT = r"D:\rs-yolo\runs\crossdomain"

# 同域基线（已知，来自 results.csv）
BASELINE = [
    {"name": "同域 val（SODA-A）· 416", "map50": 0.8228, "map50_95": 0.4080,
     "precision": 0.841, "recall": 0.771, "n_images": 698, "n_gt": 42428, "domain": "同域"},
    {"name": "同域 val（SODA-A）· 640", "map50": 0.9178, "map50_95": 0.5363,
     "precision": 0.897, "recall": 0.863, "n_images": 698, "n_gt": 42428, "domain": "同域"},
]

NICE = {
    "hrsid_sar_416_full": ("跨域 SAR（HRSID）· 416 模型", "跨模态 SAR"),
    "hrsid_sar_640": ("跨域 SAR（HRSID）· 640 模型", "跨模态 SAR"),
    "dior_val_416": ("跨域光学（DIOR）· 416 模型", "跨数据集光学"),
    "dior_val_640": ("跨域光学（DIOR）· 640 模型", "跨数据集光学"),
}


def main():
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, "*.json"))):
        key = os.path.splitext(os.path.basename(p))[0]
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        title, dom = NICE.get(key, (key, "其它"))
        rows.append({
            "name": title,
            "map50": d.get("map50"),
            "map50_95": d.get("map50_95"),
            "precision": d.get("precision"),
            "recall": d.get("recall"),
            "n_images": d.get("n_images"),
            "n_gt": d.get("n_gt"),
            "domain": dom,
            "raw": key,
        })

    all_rows = BASELINE + rows

    def fmt(v, nd=4):
        return ("%.*f" % (nd, v)) if isinstance(v, (int, float)) else "-"

    print()
    print("| 评测集 | 域 | 图片 | GT 框 | mAP50 | mAP50-95 | Precision | Recall |")
    print("|---|---|---|---|---|---|---|---|")
    for r in all_rows:
        print("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["name"], r["domain"], r["n_images"], r["n_gt"],
            fmt(r["map50"]), fmt(r["map50_95"]), fmt(r["precision"], 3), fmt(r["recall"], 3)))

    # 掉幅
    base = BASELINE[0]["map50"]
    print()
    print("相对同域 416 基线（mAP50 = %.4f）的保留率：" % base)
    for r in rows:
        if isinstance(r["map50"], (int, float)):
            print("  %-34s %6.1f%%" % (r["name"], 100.0 * r["map50"] / base))

    # 柱状图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        for fam in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
            try:
                path = font_manager.findfont(font_manager.FontProperties(family=fam),
                                             fallback_to_default=False)
                from matplotlib.font_manager import FontProperties
                matplotlib.rcParams["font.family"] = FontProperties(fname=path).get_name()
                break
            except Exception:
                continue
        matplotlib.rcParams["axes.unicode_minus"] = False

        labels = [r["name"].replace("（", "\n（") for r in all_rows]
        v50 = [r["map50"] if isinstance(r["map50"], (int, float)) else 0 for r in all_rows]
        v5095 = [r["map50_95"] if isinstance(r["map50_95"], (int, float)) else 0 for r in all_rows]
        colors = ["#3b7dd8" if r["domain"] == "同域" else ("#e8743b" if "SAR" in r["domain"] else "#2e9e5b")
                  for r in all_rows]
        fig, ax = plt.subplots(figsize=(11, 5.6), dpi=140)
        x = range(len(all_rows))
        b1 = ax.bar([i - 0.2 for i in x], v50, width=0.4, color=colors, label="mAP@0.5")
        b2 = ax.bar([i + 0.2 for i in x], v5095, width=0.4, color=colors, alpha=0.45,
                    hatch="//", label="mAP@0.5:0.95")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("精度")
        ax.set_ylim(0, 1.0)
        ax.set_title("同域 vs 跨域：模型换一个域后还剩多少精度")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        for i, v in enumerate(v50):
            ax.text(i - 0.2, v + 0.015, "%.3f" % v, ha="center", fontsize=8)
        plt.tight_layout()
        outp = os.path.join(OUT, "crossdomain_compare.png")
        plt.savefig(outp)
        print()
        print("图已保存:", outp)
    except Exception as e:
        print("画图失败:", e)


if __name__ == "__main__":
    main()
