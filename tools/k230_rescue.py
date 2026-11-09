# SPDX-FileCopyrightText: 2026 XiaoFenglistens
# SPDX-License-Identifier: MIT
"""串口救急：把卡在"半条命令"状态的板子拉回 REPL。

背景：串口传大文件时如果主机端被杀，板子会停在 raw REPL 里等剩余字节，
之后所有 REPL 工具都读不到输出。这里发换行 + Ctrl-C 把它清干净。

注意：Ctrl-C 只在**没有用户程序在跑**时安全。如果板子正在跑 main.py
（触摸启动器），Ctrl-C 会打断 finally 里的 sensor.stop() 导致摄像头被永久占用，
只能断电恢复——所以本脚本会先看有没有程序痕迹再决定发不发 Ctrl-C。
"""

import sys
import time

import serial

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM7"


def main():
    s = serial.Serial(PORT, 115200, timeout=0.6)
    time.sleep(0.2)
    s.reset_input_buffer()

    # 1) 先发一个换行，让"半条命令"自己报错退出
    s.write(b"\r\n")
    time.sleep(0.4)
    buf = s.read(4096)
    print("[1] 换行后收到 %d 字节：%r" % (len(buf), buf[-200:]))

    # 2) Ctrl-B 进 raw REPL；若板子空闲会立刻回 "raw REPL; CTRL-B to exit"
    s.write(b"\x02")
    time.sleep(0.4)
    buf2 = s.read(4096)
    print("[2] Ctrl-B 后收到 %d 字节：%r" % (len(buf2), buf2[-200:]))

    if b"raw REPL" in buf2:
        s.write(b"\x03")           # 退出 raw -> 友好 REPL
        time.sleep(0.3)
        s.read(4096)
        s.write(b"\r\n")
        time.sleep(0.3)
        print("[3] 已回到友好 REPL")
    else:
        # 没进 raw，说明可能真有程序在跑或板子没响应
        print("[3] 未进入 raw REPL，可能：①有程序在跑（请长按 USR 5 秒退出）②板子未响应")

    # 3) 最后一次探活：发一条无害语句
    s.write(b"print('PING_OK')\r\n")
    time.sleep(0.5)
    tail = s.read(4096)
    ok = b"PING_OK" in tail
    print("[4] 探活 %s：%r" % ("成功" if ok else "失败", tail[-200:]))
    s.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
