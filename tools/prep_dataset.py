"""数据集处理一键流程

目标:把 YOLO-OBB(旋转框) 的 SODA-A ship 数据集,转成可直接训练的
      水平框 YOLO 数据集,并在每步之后做校验和可视化。

为什么每步都要校验:格式合规 != 坐标正确。旋转框转正框的坐标数学
很容易出错,必须把框画回原图用眼睛确认,否则要等训练完才暴露问题。

产出结构:
    sodaa_ship_hbb/
      data.yaml          (path 写绝对路径)
      images/train,val/  复制过来的原图
      labels/train,val/  转换后的水平框标注

用法:
    python prep_dataset.py --src D:\\rs-yolo\\datasets\\sodaa_ship
    python prep_dataset.py --src <dir> --only-check      # 只校验不转换
"""
import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
SPLITS = ("train", "val")


def run(title, cmd):
    print("\n" + "=" * 70)
    print(f">>> {title}")
    print("=" * 70)
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        print(f"\n!! 这一步返回 {r.returncode},请检查上面的输出")
        return False
    return True


def copy_images(src_img, dst_img):
    """复制图片。

    注意:这里不能用 Windows junction / 符号链接。ultralytics 内部会用
    os.path.realpath() 解析路径,junction 会被穿透回源目录,导致它去读
    原始的 OBB 标签而不是我们转换后的水平框标签 —— 训练时静默用错数据。
    实测踩过这个坑,所以老老实实复制(635MB,可接受)。
    """
    if os.path.isdir(dst_img):
        n = len([f for f in os.listdir(dst_img) if f.lower().endswith(".jpg")])
        if n > 0:
            return n
    os.makedirs(dst_img, exist_ok=True)
    r = subprocess.run(
        ["robocopy", src_img, dst_img, "*.jpg",
         "/NFL", "/NDL", "/NJH", "/NJS", "/NP", "/MT:16"],
        capture_output=True, text=True,
    )
    if not os.path.isdir(dst_img):
        return 0
    return len([f for f in os.listdir(dst_img) if f.lower().endswith(".jpg")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="原始 OBB 数据集目录(含 data.yaml)")
    ap.add_argument("--dst", default=None, help="输出数据集目录(默认 <src>_hbb)")
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--skip-obb-vis", action="store_true")
    ap.add_argument("--only-check", action="store_true")
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst) if args.dst else src + "_hbb"
    data_yaml = os.path.join(src, "data.yaml")
    if not os.path.exists(data_yaml):
        print(f"找不到 {data_yaml}")
        sys.exit(1)

    print("原始数据集:", src)
    print("输出数据集:", dst)

    # ---- 1. 校验原始 OBB ----
    run("步骤 1/6  校验原始 OBB 标注格式",
        [PY, "check_dataset.py", "--data", data_yaml,
         "--lists", "train", "val", "--no-vis"])

    # ---- 2. 可视化原始 OBB ----
    if not args.skip_obb_vis:
        run("步骤 2/6  可视化原始旋转框(确认标注本身没问题)",
            [PY, "check_dataset.py", "--data", data_yaml,
             "--lists", "train", "--samples", str(args.samples),
             "--out", os.path.join(src, "_vis_obb")])

    if args.only_check:
        print("\n--only-check:结束于校验步骤")
        return

    # ---- 3. 搭 HBB 数据集(复制图片) ----
    print("\n" + "=" * 70)
    print(">>> 步骤 3/6  搭建 HBB 数据集目录(复制图片)")
    print("=" * 70)
    os.makedirs(os.path.join(dst, "labels"), exist_ok=True)
    for sp in SPLITS:
        src_img = os.path.join(src, "images", sp)
        if not os.path.isdir(src_img):
            print(f"  跳过 {sp}: 原图目录不存在 {src_img}")
            continue
        n = copy_images(src_img, os.path.join(dst, "images", sp))
        print(f"  images/{sp}: {n} 个文件")
        os.makedirs(os.path.join(dst, "labels", sp), exist_ok=True)

    # ---- 4. 转换 OBB -> HBB ----
    run("步骤 4/6  OBB -> 水平框 转换(dry-run 统计)",
        [PY, "obb_to_hbb.py", "--src", os.path.join(src, "labels"), "--dry-run",
         "--splits", *SPLITS])

    if not run("步骤 4b/6  写入水平框标注",
               [PY, "obb_to_hbb.py", "--src", os.path.join(src, "labels"),
                "--dst", os.path.join(dst, "labels"), "--splits", *SPLITS]):
        return

    # ---- 生成 data.yaml ----
    # 注意:ultralytics 把 path 当「相对当前工作目录」解析,不是相对 yaml 文件,
    # 所以这里必须写绝对路径(正斜杠,兼容 Windows)。
    hbb_yaml = os.path.join(dst, "data.yaml")
    root_posix = dst.replace("\\", "/")
    with open(hbb_yaml, "w", encoding="utf-8") as f:
        f.write(f"path: {root_posix}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write("names:\n  0: ship\n")
    print("\n已生成:", hbb_yaml)

    # ---- 5. 校验 HBB ----
    run("步骤 5/6  校验转换后的水平框格式",
        [PY, "check_dataset.py", "--data", hbb_yaml,
         "--lists", "train", "val", "--no-vis"])

    # ---- 6. 可视化 HBB ----
    run("步骤 6/6  可视化水平框(最关键的一步,请仔细看图)",
        [PY, "check_dataset.py", "--data", hbb_yaml,
         "--lists", "train", "--samples", str(args.samples),
         "--out", os.path.join(dst, "_vis_hbb")])

    print("\n" + "=" * 70)
    print("全部完成。请打开这两张对比图,肉眼确认:")
    print("  原始旋转框:", os.path.join(src, "_vis_obb"))
    print("  转换后正框:", os.path.join(dst, "_vis_hbb"))
    print("重点看:正框是否正好包住船体、有没有框到大片空白/岸边。")
    print("\n确认无误后开始训练:")
    print(f'  D:\\rs-yolo\\venv\\Scripts\\yolo.exe task=detect mode=train '
          f'model=yolov8n.pt data="{hbb_yaml}" epochs=100 imgsz=320 batch=16')
    print("=" * 70)


if __name__ == "__main__":
    main()
