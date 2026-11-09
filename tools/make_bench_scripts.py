# SPDX-FileCopyrightText: 2026 XiaoFenglistens
# SPDX-License-Identifier: MIT
"""从 bench5_512.py 生成指定 kmodel 的测速脚本（避免手改 print 格式串出错）。"""

import ast
import os

SRC = r"D:\rs-yolo\tmp\bench5_512.py"
OUT_DIR = r"D:\rs-yolo\tmp"

VARIANTS = {
    "i8": "/sdcard/best512.kmodel",
    "a16": "/sdcard/best512_a16.kmodel",
    "i16": "/sdcard/best512_i16.kmodel",
}


def build(tag, kmodel):
    src = open(SRC, encoding="utf-8").read()
    src = src.replace("/sdcard/bench512.kmodel", kmodel)
    # 只给日志加标签，保持占位符与参数一一对应（不动格式串结构）
    src = src.replace('print("[B] ready imgsz=', 'print("[B] ready ' + tag + ' imgsz=')
    src = src.replace('print("[B] imgsz=%d frames=%d', 'print("[B] ' + tag + ' imgsz=%d frames=%d')
    ast.parse(src)  # 语法自检
    out = os.path.join(OUT_DIR, "bench_%s_512.py" % tag)
    open(out, "w", encoding="utf-8").write(src)
    for line in src.splitlines():
        if "[B] ready" in line or "[B] " + tag + " imgsz" in line:
            print("   ", line.strip())
    print("wrote", out, len(src), "chars")


if __name__ == "__main__":
    for tag, km in VARIANTS.items():
        build(tag, km)
