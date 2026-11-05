# SPDX-FileCopyrightText: 2026 XiaoFenglistens
# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""
板端测速编排：把 4 档输入尺寸的 bench 脚本跑一遍，收集 fps / 单帧推理耗时。

前提：
  1) 板子已退出采集程序（长按 USR 3 秒），串口空闲
  2) 416 / 640 的 kmodel 已在 /sdcard（best.kmodel / best640.kmodel）
  3) 320 / 512 的 kmodel 由本脚本自动上传

用法：
  python tools\\pareto_board.py                # 跑全部四档
  python tools\\pareto_board.py --only 320 416 # 只跑指定档
"""
import os as _os
KROOT = _os.environ.get("K230_ROOT", ".")  # 项目根目录，默认当前目录
import argparse
import json
import os
import re
import subprocess
import sys

PY = _os.path.join(KROOT, 'venv', 'Scripts', 'python.exe')
REPL = _os.path.join(KROOT, 'tools', 'k230_repl.py')
TMP = _os.path.join(KROOT, 'tmp')
RUNS = _os.path.join(KROOT, 'runs')

# imgsz -> (本地 bench 脚本, 板上 kmodel, 本地 kmodel 来源<可能为 None>, 是否需上传)
PLAN = {
    320: ("bench320.py", "/sdcard/bench320.kmodel",
          _os.path.join(KROOT, 'runs', 'ship_320_v1', 'weights', 'best.kmodel')),
    416: ("bench416.py", "/sdcard/best.kmodel", None),
    512: ("bench512.py", "/sdcard/bench512.kmodel",
          _os.path.join(KROOT, 'runs', 'ship_512_v1', 'weights', 'best.kmodel')),
    640: ("bench640.py", "/sdcard/best640.kmodel", None),
}

LINE = re.compile(r"\[B\] imgsz=(\d+).*?fps=([\d.]+).*?infer_mean=([\d.]+)ms")


def sh(args, timeout=1800):
    print("  $ %s" % " ".join(args[1:]))
    r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", type=int, default=None)
    args = ap.parse_args()
    sizes = args.only or sorted(PLAN)

    out_path = os.path.join(RUNS, "pareto_board.json")
    board = {}
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            board = json.load(f)

    for s in sizes:
        local_bench, remote_km, local_km = PLAN[s]
        print("\n===== imgsz=%d =====" % s)

        # 1) kmodel
        if local_km:
            if not os.path.exists(local_km):
                print("  [skip] 缺少 kmodel: %s" % local_km)
                continue
            mb = os.path.getsize(local_km) / 1048576.0
            print("  上传 kmodel %.2f MB（约 %.1f 分钟 @17KB/s）" % (mb, mb * 1024 / 17.0 / 60))
            rc, o = sh([PY, REPL, "put", local_km, remote_km, "1024"], timeout=1800)
            if rc != 0 or "成功=True" not in o:
                print("  [fail] kmodel 上传失败：%s" % o.strip()[-300:])
                continue
        else:
            mb = None

        # 2) bench 脚本
        bp = os.path.join(TMP, local_bench)
        rc, o = sh([PY, REPL, "put", bp, "/sdcard/" + local_bench, "1024"], timeout=120)
        if rc != 0:
            print("  [fail] 脚本上传失败：%s" % o.strip()[-300:])
            continue

        # 3) 跑
        rc, o = sh([PY, REPL, "run", "/sdcard/" + local_bench, "600"], timeout=900)
        m = LINE.search(o)
        if not m:
            print("  [fail] 未解析到结果，原始输出尾部：\n%s" % o.strip()[-800:])
            continue
        fps = float(m.group(2))
        infer = float(m.group(3))
        board[str(s)] = {"fps": round(fps, 1), "infer_ms": round(infer, 1),
                         "kmodel_mb": round(mb, 2) if mb else None}
        print("  -> fps=%.1f  infer=%.1fms" % (fps, infer))

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(board, f, ensure_ascii=False, indent=2)

    print("\n===== 汇总（%s）=====" % out_path)
    for k in sorted(board, key=int):
        v = board[k]
        print("  imgsz=%-4s fps=%-6.1f infer=%-6.1fms kmodel=%s MB"
              % (k, v["fps"], v["infer_ms"], v["kmodel_mb"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
