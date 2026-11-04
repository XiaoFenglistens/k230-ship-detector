# SPDX-FileCopyrightText: 2026 XiaoFenglistens
# SPDX-License-Identifier: MIT
"""YOLO 数据集校验 + 可视化

用途:训练前确认标注格式正确、坐标落在正确位置。

为什么必须做:标注文件里 "0.51 0.48 0.12 0.09" 这种数字,
格式合规不代表框画对了。DOTA/旋转框转水平框时最容易出错,
而这个错误只有把框画回原图才能发现,否则要等训练完才暴露。

用法:
    python check_dataset.py --data <data.yaml> [--samples 12] [--out vis]
    python check_dataset.py --data data.yaml --lists train val
"""
import argparse
import os
import random
import sys
from collections import Counter

try:
    import yaml
except ImportError:
    print("需要 pyyaml: pip install pyyaml")
    sys.exit(1)

try:
    from PIL import Image, ImageDraw
except ImportError:
    print("需要 pillow: pip install pillow")
    sys.exit(1)


# ----------------------------- 基础工具 -------------------------------------
def find_split_dirs(data_yaml_path, cfg):
    """从 data.yaml 里解析出各 split 的图片目录。"""
    base = os.path.dirname(os.path.abspath(data_yaml_path))
    splits = {}
    for key in ("train", "val", "test"):
        val = cfg.get(key)
        if not val:
            continue
        paths = val if isinstance(val, list) else [val]
        for p in paths:
            full = p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))
            if os.path.isdir(full):
                splits[key] = full
                break
    return splits


def label_path_for(img_path, label_root=None):
    """由 images/xxx.jpg 推出对应的 labels/xxx.txt。

    label_root 不为空时,强制把图片路径映射到该标注根目录下,
    这样就能对同一批图片分别校验不同的标签版本
    (例如原始 OBB 与转换后的 HBB 做对比)。
    """
    norm = os.path.normpath(img_path)
    if label_root:
        # 取 images/<split>/<name>.jpg 里 <split>/<name> 部分
        parts = norm.split(os.sep)
        try:
            idx = parts.index("images")
            tail = parts[idx + 1:]
        except ValueError:
            tail = [parts[-1]]
        return os.path.join(label_root, *tail).rsplit(".", 1)[0] + ".txt"

    parts = norm.split(os.sep)
    for i, seg in enumerate(parts):
        if seg == "images":
            parts[i] = "labels"
            break
    return os.sep.join(parts).rsplit(".", 1)[0] + ".txt"


def parse_label_line(line, ncoords):
    """解析一行标注,返回 (class_id, coords...) 或 None。"""
    f = line.split()
    if len(f) != ncoords + 1:
        return None
    try:
        cid = int(float(f[0]))
        coords = [float(v) for v in f[1:]]
    except ValueError:
        return None
    return cid, coords


# ----------------------------- 校验逻辑 -------------------------------------
def validate_split(name, img_dir, class_names, ncoords, label_root=None, verbose=False):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    images = sorted(
        f for f in os.listdir(img_dir) if f.lower().endswith(exts)
    )
    problems = Counter()
    bad_examples = []
    class_counts = Counter()
    obj_total = 0
    missing_label = 0
    empty_label = 0
    size_mismatch = 0
    auto_ncoords = Counter()

    for fn in images:
        ipath = os.path.join(img_dir, fn)
        lpath = label_path_for(ipath, label_root)

        if not os.path.exists(lpath):
            missing_label += 1
            if len(bad_examples) < 5:
                bad_examples.append(f"缺标注文件: {fn}")
            continue

        with open(lpath) as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        if not lines:
            empty_label += 1
            continue

        # 图片尺寸用于还原像素坐标(仅用于报告)
        try:
            with Image.open(ipath) as im:
                W, H = im.size
        except Exception as e:
            problems["图片无法打开"] += 1
            if len(bad_examples) < 5:
                bad_examples.append(f"无法打开: {fn} ({e})")
            continue

        for ln in lines:
            # 自动识别每行的字段数(兼容 OBB 9 段与 HBB 5 段混杂的情况)
            nf = len(ln.split())
            auto_ncoords[nf - 1] += 1
            parsed = parse_label_line(ln, nf - 1)
            if parsed is None:
                problems[f"字段数错误(应为{ncoords + 1})"] += 1
                if len(bad_examples) < 5:
                    bad_examples.append(f"{fn}: '{ln[:60]}'")
                continue

            cid, coords = parsed
            obj_total += 1
            class_counts[cid] += 1

            if cid < 0 or cid >= len(class_names):
                problems["类别 id 越界"] += 1
                if len(bad_examples) < 5:
                    bad_examples.append(f"{fn}: 类别 id={cid} (共 {len(class_names)} 类)")

            if any(c < 0 or c > 1 for c in coords):
                problems["坐标超出 0~1 范围"] += 1
                if len(bad_examples) < 5:
                    bad_examples.append(f"{fn}: 坐标越界 {[round(c, 3) for c in coords][:4]}")

            # 正框额外检查:宽高必须为正
            if len(coords) == 4:
                w, h = coords[2], coords[3]
                if w <= 0 or h <= 0:
                    problems["宽或高 <= 0"] += 1
                    if len(bad_examples) < 5:
                        bad_examples.append(f"{fn}: w={w:.4f} h={h:.4f}")

    print(f"\n[{name}] {img_dir}")
    if label_root:
        print(f"  标注来自      : {label_root}")
    print(f"  图片数        : {len(images)}")
    print(f"  标注实例数    : {obj_total}")
    print(f"  缺标注文件    : {missing_label}")
    print(f"  空标注文件    : {empty_label}")
    fmt = ", ".join(f"{k}坐标" for k in sorted(auto_ncoords) if k > 0)
    if fmt:
        kind = "OBB 旋转框" if 8 in auto_ncoords else "水平框"
        print(f"  实际字段数    : {fmt}  -> 判定为 {kind}")
    if not class_names:
        print("  (data.yaml 未提供 names)")
    else:
        print("  各类别实例数  :")
        for cid in sorted(class_counts):
            nm = class_names[cid] if cid < len(class_names) else f"<未知{cid}>"
            print(f"      {cid} {nm:<20} {class_counts[cid]}")
    if problems:
        print("  !! 发现格式问题:")
        for k, v in problems.most_common():
            print(f"      {k}: {v}")
        for ex in bad_examples:
            print(f"      例: {ex}")
    else:
        print("  格式检查: 通过")
    return obj_total, class_counts, problems


# ----------------------------- 可视化 ---------------------------------------
def visualize_split(name, img_dir, class_names, ncoords, out_dir, n_samples=12,
                    label_root=None):
    """把标注框画回原图,人工确认位置是否正确。"""
    os.makedirs(out_dir, exist_ok=True)
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    images = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(exts))
    if not images:
        print(f"[{name}] 没有图片,跳过可视化")
        return 0

    random.seed(0)
    picked = random.sample(images, min(n_samples, len(images)))
    drawn = 0

    for fn in picked:
        ipath = os.path.join(img_dir, fn)
        lpath = label_path_for(ipath, label_root)
        if not os.path.exists(lpath):
            continue

        with Image.open(ipath) as im:
            im = im.convert("RGB")
            W, H = im.size
            # 原图可能很大(遥感图 4000x4000),缩到可读尺寸
            scale = min(1.0, 900.0 / max(W, H))
            if scale < 1.0:
                im = im.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
            d = ImageDraw.Draw(im)

            with open(lpath) as f:
                for ln in f:
                    ln = ln.strip()
                    nf = len(ln.split())
                    parsed = parse_label_line(ln, nf - 1)
                    if parsed is None:
                        continue
                    cid, coords = parsed

                    if len(coords) == 4:
                        cx, cy, bw, bh = coords
                        x1 = (cx - bw / 2) * W * scale
                        y1 = (cy - bh / 2) * H * scale
                        x2 = (cx + bw / 2) * W * scale
                        y2 = (cy + bh / 2) * H * scale
                        d.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
                        tx, ty = x1, max(0, y1 - 14)
                    else:
                        # OBB: 4 个点 (x1,y1,x2,y2,x3,y3,x4,y4)
                        pts = [
                            (coords[i] * W * scale, coords[i + 1] * H * scale)
                            for i in range(0, 8, 2)
                        ]
                        d.polygon(pts, outline=(255, 0, 0))
                        for i in range(4):
                            d.line([pts[i], pts[(i + 1) % 4]], fill=(255, 0, 0), width=3)
                        tx, ty = pts[0][0], max(0, pts[0][1] - 14)

                    nm = class_names[cid] if cid < len(class_names) else str(cid)
                    d.text((tx + 2, ty), nm, fill=(255, 255, 0))

        out_path = os.path.join(out_dir, f"{name}_{fn.rsplit('.', 1)[0]}_vis.jpg")
        im.save(out_path, quality=88)
        drawn += 1

    print(f"[{name}] 可视化 {drawn} 张 -> {out_dir}")
    return drawn


# ----------------------------- 主流程 ---------------------------------------
def main():
    ap = argparse.ArgumentParser(description="YOLO 数据集校验与可视化")
    ap.add_argument("--data", required=True, help="data.yaml 路径")
    ap.add_argument("--samples", type=int, default=12, help="每个 split 可视化几张")
    ap.add_argument("--out", default=None, help="可视化输出目录(默认 <数据集目录>/_vis)")
    ap.add_argument("--lists", nargs="*", default=None, help="只检查指定 split,如 train val")
    ap.add_argument("--label-root", default=None,
                    help="强制从该目录读标注(用于对比不同标签版本,如 labels_hbb)")
    ap.add_argument("--no-vis", action="store_true", help="只做格式校验,不画图")
    args = ap.parse_args()

    with open(args.data, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    class_names = cfg.get("names") or []
    if isinstance(class_names, dict):
        class_names = [class_names[k] for k in sorted(class_names)]

    ncoords = 4
    note = "水平框 (cx cy w h)"
    if cfg.get("obb") or cfg.get("task") == "obb":
        ncoords = 8
        note = "旋转框 OBB (4 点)"

    print("=" * 62)
    print("数据集校验报告")
    print("=" * 62)
    print(f"data.yaml : {args.data}")
    print(f"类别数    : {len(class_names)}")
    print(f"类别名称  : {class_names}")
    print(f"标注格式  : {note}")

    splits = find_split_dirs(args.data, cfg)
    if not splits:
        print("\n!! data.yaml 里的 train/val/test 路径都不存在,检查路径是否正确")
        sys.exit(1)
    print(f"发现 split: {list(splits.keys())}")

    if args.lists:
        splits = {k: v for k, v in splits.items() if k in args.lists}

    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(args.data)), "_vis")

    total_problems = Counter()
    for name, d in splits.items():
        _, _, probs = validate_split(name, d, class_names, ncoords,
                                     label_root=args.label_root)
        total_problems.update(probs)
        if not args.no_vis:
            visualize_split(name, d, class_names, ncoords, out_dir, args.samples,
                            label_root=args.label_root)

    print("\n" + "=" * 62)
    if total_problems:
        print(f"结论: 发现 {sum(total_problems.values())} 处格式问题,建议先修数据")
    else:
        print("结论: 格式校验全部通过")
    if not args.no_vis:
        print(f"请打开 {out_dir} 里的图片,肉眼确认检测框框住了目标")
        print("(格式正确 != 位置正确,这一步不能省)")
    print("=" * 62)


if __name__ == "__main__":
    main()
