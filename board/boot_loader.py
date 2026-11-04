# -*- coding: utf-8 -*-
# 开机启动引导：运行 /sdcard/main.py
# 用法：在 CanMV IDE 里打开 /sdcard/boot.py，全选删除，粘贴本文件全部内容，保存。
import sys

try:
    with open("/sdcard/main.py") as _f:
        _src = _f.read()
    exec(_src, {"__name__": "__main__"})
except Exception as _e:
    print("boot.py 启动 main.py 失败:", _e)
    sys.print_exception(_e)
