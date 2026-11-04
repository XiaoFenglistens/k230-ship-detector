"""YOLO-OBB(旋转框) -> YOLO 水平框(axis-aligned) 转换

背景:
    SODA-A ship 数据集是 YOLO-OBB 格式,每行 9 个字段:
        class  x1 y1  x2 y2  x3 y3  x4 y4     (4 个角点,已归一化 0~1)
    而 K230 上部署链路最成熟的是普通检测(正框),每行 5 个字段:
        class  cx cy w h                      (中心点 + 宽高,归一化)

做法:
    把旋转框的外接四边形取「最小外接水平矩形」(轴对齐包围盒):
        xmin = min(x1..x4), xmax = max(x1..x4)
        cx = (xmin+xmax)/2,  w = xmax-xmin,   y 同理
    斜向目标外接框会略大,但部署简单、官方例程直接可用。

用法:
    # 先看统计,不写文件(强烈建议第一步)
    python obb_to_hbb.py --src labels --dry-run

    # 实际转换,输出到 labels_hbb/
    python obb_to_hbb.py --src labels --dst labels_hbb

    # 只转 train/val(不转 test)
    python obb_to_hbb.py --src labels --dst labels_hbb --splits train val
"""
import argparse
import os
import shutil
from collections import Counter


def parse_obb_line(line):
    """解析一行 OBB 标注 -> (cid, [(x,y) x4]) 或 None"""
    f = line.split()
    if len(f) != 9:
        return None
    try:
        cid = int(float(f[0]))
        vals = [float(v) for v in f[1:]]
    except ValueError:
        return None
    pts = [(vals[i], vals[i + 1]) for i in range(0, 8, 2)]
    return cid, pts


def obb_to_hbb(pts, min_size=1e-3):
    """4 个角点 -> (cx, cy, w, h),归一化坐标。"""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    # 裁到合法范围:标注偶尔会有轻微越界
    xmin = max(0.0, min(1.0, xmin))
    xmax = max(0.0, min(1.0, xmax))
    ymin = max(0.0, min(1.0, ymin))
    ymax = max(0.0, min(1.0, ymax))

    w = xmax - xmin
    h = ymax - ymin
    if w < min_size or h < min_size:
        return None

    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    return cx, cy, w, h


def convert_dir(src_dir, dst_dir, min_size, dry_run, stats):
    if not dry_run:
        os.makedirs(dst_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(src_dir) if f.endswith(".txt"))
    if not files:
        return 0

    for fn in files:
        sp = os.path.join(src_dir, fn)
        out_lines = []
        with open(sp) as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                stats["输入行数"] += 1
                parsed = parse_obb_line(ln)
                if parsed is None:
                    stats["字段数异常(跳过)"] += 1
                    continue
                cid, pts = parsed

                # 面积退化 / 越界检查
                if any(not (0 <= v <= 1) for p in pts for v in p):
                    stats["原始坐标越界"] += 1

                hbb = obb_to_hbb(pts, min_size)
                if hbb is None:
                    stats["退化框(跳过)"] += 1
                    continue

                cx, cy, w, h = hbb
                out_lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                stats["输出框数"] += 1

                # 顺带统计长宽比,识别异常细长的框
                if w > 0 and h > 0:
                    ar = max(w / h, h / w)
                    if ar > 10:
                        stats["长宽比>10(可疑)"] += 1

        if not dry_run:
            with open(os.path.join(dst_dir, fn), "w") as f:
                f.write("\n".join(out_lines) + ("\n" if out_lines else ""))

    return len(files)


def main():
    ap = argparse.ArgumentParser(description="YOLO-OBB -> YOLO 水平框")
    ap.add_argument("--src", required=True, help="OBB 标注根目录(含 train/val/test)")
    ap.add_argument("--dst", default=None, help="输出根目录")
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--min-size", type=float, default=1e-3,
                    help="小于此归一化尺寸的框丢弃(默认 0.001)")
    ap.add_argument("--dry-run", action="store_true", help="只统计不写文件")
    args = ap.parse_args()

    print("=" * 62)
    print("YOLO-OBB -> 水平框 转换")
    print("=" * 62)
    print(f"源目录  : {args.src}")
    print(f"输出目录: {args.dst if args.dst else '(dry-run 不输出)'}")
    print(f"处理 split: {args.splits}")
    print(f"最小尺寸阈值: {args.min_size}")
    print("=" * 62)

    grand = Counter()
    for sp in args.splits:
        src = os.path.join(args.src, sp)
        if not os.path.isdir(src):
            print(f"\n[{sp}] 跳过(目录不存在: {src})")
            continue
        dst = os.path.join(args.dst, sp) if args.dst else None
        stats = Counter()
        n = convert_dir(src, dst, args.min_size, args.dry_run, stats)

        print(f"\n[{sp}] {n} 个标注文件 -> {dst or '(dry-run)'}")
        for k in ("输入行数", "输出框数", "字段数异常(跳过)", "退化框(跳过)",
                  "原始坐标越界", "长宽比>10(可疑)"):
            if stats.get(k):
                print(f"    {k:<20} {stats[k]}")
        grand.update(stats)

    print("\n" + "=" * 62)
    print("合计:")
    for k in ("输入行数", "输出框数", "字段数异常(跳过)", "退化框(跳过)",
              "原始坐标越界", "长宽比>10(可疑)"):
        if grand.get(k):
            print(f"    {k:<20} {grand[k]}")
    dropped = grand.get("字段数异常(跳过)", 0) + grand.get("退化框(跳过)", 0)
    if dropped:
        print(f"\n注意: 丢弃了 {dropped} 个框,建议核对是否为数据问题")
    else:
        print("\n没有丢弃任何框")
    print("=" * 62)
    if args.dry_run:
        print("这是 dry-run,未写入任何文件。确认无误后去掉 --dry-run 再跑。")
    else:
        print("下一步:用 check_dataset.py 可视化校验转换结果,确认框位置正确")
    print("=" * 62)


if __name__ == "__main__":
    main()
