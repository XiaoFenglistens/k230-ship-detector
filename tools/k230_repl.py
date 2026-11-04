# -*- coding: utf-8 -*-
"""极简 K230 MicroPython REPL 客户端（走 USB CDC 串口）。

用法：
    python k230_repl.py                 # 交互探测：hello + 列 /sdcard
    python k230_repl.py cmd "<code>"    # 执行一段代码并打印输出
    python k230_repl.py ls [路径]        # 列目录
    python k230_repl.py raw <文件>       # 用 paste 模式把本地文件传到板子内存执行

注意：CanMV IDE 打开着串口时本脚本会拿不到句柄（PermissionError），
      需要先在 IDE 里断开连接。
"""
import sys
import time

import serial

PORT = "COM7"
BAUD = 115200


class Repl:
    def __init__(self, port=PORT, baud=BAUD):
        self.ser = serial.Serial(port, baud, timeout=0.2)
        self.ser.reset_input_buffer()

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass

    def drain(self, wait=0.25):
        """把当前缓冲区里的字节全部读出来。"""
        time.sleep(wait)
        out = b""
        idle = 0
        while idle < 3:
            n = self.ser.in_waiting
            if n:
                out += self.ser.read(n)
                idle = 0
                time.sleep(0.02)
            else:
                idle += 1
                time.sleep(0.02)
        return out

    def interrupt(self):
        """Ctrl-C：中断正在运行的程序，回到 REPL 提示符。"""
        for _ in range(2):
            self.ser.write(b"\x03")
            time.sleep(0.15)
        return self.drain(0.3)

    def friendly(self):
        """Ctrl-B：普通（友好）REPL，带 >>> 回显。"""
        self.ser.write(b"\x02")
        time.sleep(0.15)
        return self.drain(0.2)

    def exec_line(self, code, wait=0.35):
        """执行一行代码。多行代码请用 exec('...') 包一层或改用 exec_block。"""
        self.ser.write(code.encode("utf-8") + b"\r\n")
        return self.drain(wait)

    def exec_block(self, code, wait=1.0):
        """执行多行代码：用 exec(<repr>) 单行提交，避免 REPL 缩进问题。

        必须给 exec 一个**全新**的 globals 字典：raw REPL 默认在 __main__ 的
        全局命名空间里执行，上一轮留下的 sensor/image 等大对象会被一直引用，
        gc 收不掉，跑几次就 OSError ENOMEM。
        """
        one_line = "exec(" + repr(code) + ", {'__name__': '__main__'})"
        return self.exec_line(one_line, wait)

    # ---------------- raw REPL（用于可靠地传文件） ----------------

    def raw_enter(self):
        """Ctrl-A：进入 raw REPL。"""
        self.ser.write(b"\r\x01")
        time.sleep(0.08)
        return self.drain(0.12)

    def raw_leave(self):
        """Ctrl-B：退出 raw REPL，回到友好 REPL。"""
        self.ser.write(b"\r\x02")
        time.sleep(0.08)
        return self.drain(0.12)

    def raw_exec(self, code, timeout=5.0):
        """raw REPL 执行：代码 + Ctrl-D。

        设备返回格式是 OK + stdout + \\x04 + stderr(traceback) + \\x04，
        也就是**两个**结束标记。只读一个会把 traceback 丢掉，所以必须等两个。
        """
        self.ser.write(code.encode("utf-8") + b"\x04")
        buf = b""
        t0 = time.time()
        while time.time() - t0 < timeout:
            n = self.ser.in_waiting
            if n:
                buf += self.ser.read(n)
                if buf.count(b"\x04") >= 2:
                    time.sleep(0.01)
                    if self.ser.in_waiting:
                        buf += self.ser.read(self.ser.in_waiting)
                    break
            else:
                time.sleep(0.003)
        return buf

    def write_file(self, local, remote, chunk=1024, verbose=True):
        """把本地文件写到板子上。

        用 bytes.fromhex() 而不是 bytes 字面量：hex 串是纯 ASCII，
        不涉及引号/反斜杠转义，raw REPL 逐字节透传最稳。
        返回 (耗时秒, 是否成功, 板子返回的错误)。
        """
        data = open(local, "rb").read()
        total = len(data)
        self.raw_enter()
        out = self.raw_exec("f=open(%r,'wb')" % remote)
        if b"Traceback" in out:
            self.raw_leave()
            return 0.0, False, clean(out)[:400]
        t0 = time.time()
        done = 0
        err = ""
        for i in range(0, total, chunk):
            blk = data[i : i + chunk]
            out = self.raw_exec("f.write(bytes.fromhex('%s'))" % blk.hex())
            done += len(blk)
            if b"Traceback" in out:
                err = clean(out)[:500]
                break
            if verbose and (i // chunk) % 64 == 0:
                el = time.time() - t0
                print(
                    "   %5.1f%%  %8d / %d  %6.1f s  %5.0f KB/s"
                    % (100.0 * done / total, done, total, el, done / 1024.0 / el if el else 0)
                )
        self.raw_exec("f.close()")
        el = time.time() - t0
        self.raw_leave()
        return el, (err == ""), err

    def read_file(self, remote, timeout=10.0, verbose=False):
        """把板子上的文件读回来（raw REPL 分块读）。

        每块 1024 字节：raw REPL 的行长度上限约 4000 字符，
        print(hex) 会翻倍成 2048 字符，安全。
        """
        self.raw_enter()
        self.raw_exec("f=open(%r,'rb')" % remote)
        out = b""
        t0 = time.time()
        nblk = 0
        while True:
            r = self.raw_exec("print(f.read(1024).hex())", timeout=timeout)
            # 去掉 OK 前缀与结束标记
            body = r.replace(b"OK", b"", 1).split(b"\x04")[0]
            text = body.decode("utf-8", "replace").strip()
            if not text:
                break
            try:
                chunk = bytes.fromhex(text)
            except ValueError:
                break
            if not chunk:
                break
            out += chunk
            nblk += 1
            if verbose and nblk % 32 == 0:
                el = time.time() - t0
                print(
                    "   %8d bytes  %6.1f s  %5.0f KB/s"
                    % (len(out), el, len(out) / 1024.0 / el if el else 0)
                )
        self.raw_exec("f.close()")
        self.raw_leave()
        return out



def clean(raw):
    """去掉 REPL 回显里的控制码，转成可读文本。"""
    text = raw.decode("utf-8", "replace")
    out = []
    for ch in text:
        if ch == "\r":
            continue
        if ch == "\x08":  # 退格
            if out:
                out.pop()
            continue
        out.append(ch)
    return "".join(out)


def show(title, raw):
    print("---------- %s ----------" % title)
    body = clean(raw).strip("\n")
    print(body if body else "（无输出）")
    print()


def main():
    args = sys.argv[1:]
    r = Repl()
    try:
        show("开机后缓冲区（板子可能在跑程序）", r.drain(0.6))
        show("Ctrl-C 中断", r.interrupt())
        show("进入友好 REPL", r.friendly())

        if not args:
            show("hello 测试", r.exec_line('print("PC-REPL-OK")'))
            show(
                "板子信息",
                r.exec_block(
                    "import os,sys,gc\n"
                    "print('impl', sys.implementation)\n"
                    "print('mem_free_kb', gc.mem_free()//1024)\n"
                    "print('cwd', os.getcwd())\n"
                    "print('root', sorted(os.listdir('/')))\n"
                    "for mp in ('/sdcard','/data','/flash'):\n"
                    "    try:\n"
                    "        print(mp, sorted(os.listdir(mp)))\n"
                    "    except Exception as e:\n"
                    "        print(mp, 'ERR', e)\n",
                    wait=1.5,
                ),
            )
            return

        cmd = args[0]
        if cmd == "speed":
            # 测速：往 /sdcard 写 256KB 随机数据（不保留成品，测完删掉）
            import os as _os
            import tempfile

            tmp = _os.path.join(r"D:\rs-yolo\tmp", "_speedtest.bin")
            _os.makedirs(r"D:\rs-yolo\tmp", exist_ok=True)
            blob = bytes(bytearray(range(256))) * 1024  # 256KB
            open(tmp, "wb").write(blob)
            print("测试文件 %d 字节，chunk=%s" % (len(blob), args[1] if len(args) > 1 else "4096"))
            el, ok, err = r.write_file(tmp, "/sdcard/_speedtest.bin", int(args[1]) if len(args) > 1 else 4096)
            print(
                "\n结果: %.1f s  %.1f KB/s  成功=%s  %s"
                % (el, len(blob) / 1024.0 / el if el else 0, ok, err)
            )
            show("清理", r.exec_block("import os\ntry:\n    os.remove('/sdcard/_speedtest.bin')\n    print('removed')\nexcept Exception as e:\n    print('err', e)\n"))
        elif cmd == "put":
            local, remote = args[1], args[2]
            chunk = int(args[3]) if len(args) > 3 else 1024
            size = __import__("os").path.getsize(local)
            print("上传 %s -> %s  (%d 字节, chunk=%d)" % (local, remote, size, chunk))
            el, ok, err = r.write_file(local, remote, chunk)
            print(
                "\n完成: %.1f s  %.1f KB/s  成功=%s  %s"
                % (el, size / 1024.0 / el if el else 0, ok, err)
            )
        elif cmd == "cleanup":
            # 上次异常退出后 sensor/display 可能没释放，这里强制回收
            code = (
                "import gc\n"
                "try:\n"
                "    import media.media as _m\n"
                "    _m.MediaManager.deinit()\n"
                "    print('media deinit ok')\n"
                "except Exception as e:\n"
                "    print('media deinit:', e)\n"
                "try:\n"
                "    from media.sensor import Sensor\n"
                "    Sensor.deinit()\n"
                "    print('sensor deinit ok')\n"
                "except Exception as e:\n"
                "    print('sensor deinit:', e)\n"
                "try:\n"
                "    from machine import Display\n"
                "    Display.deinit()\n"
                "    print('display deinit ok')\n"
                "except Exception as e:\n"
                "    print('display deinit:', e)\n"
                "gc.collect()\n"
                "print('free_kb', gc.mem_free()//1024)\n"
            )
            show("清理资源", r.exec_block(code, wait=2.5))
        elif cmd == "ls":
            path = args[1] if len(args) > 1 else "/sdcard"
            code = (
                "import os\n"
                "p=" + repr(path) + "\n"
                "for n in sorted(os.listdir(p)):\n"
                "    fp=p+'/'+n\n"
                "    try:\n"
                "        st=os.stat(fp)\n"
                "        print('%10d  %s%s' % (st[6], n, '/' if st[0]&0x4000 else ''))\n"
                "    except Exception as e:\n"
                "        print('  ?  ', n, e)\n"
            )
            show("ls %s" % path, r.exec_block(code, wait=1.5))
        elif cmd == "run":
            # 跑板子上的脚本文件，stdout 和 traceback 都带回来
            remote = args[1]
            timeout = float(args[2]) if len(args) > 2 else 90.0
            r.raw_enter()
            # 先读后关：exec(open(...).read()) 会泄漏文件句柄，
            # 导致后续 put 覆盖同名文件时报 OSError EPERM
            # 且必须用全新 globals，否则脚本里的 sensor/image 常驻内存 -> ENOMEM
            code = (
                "f=open(%r)\n"
                "s=f.read()\n"
                "f.close()\n"
                "try:\n"
                "    exec(s, {'__name__': '__main__'})\n"
                "finally:\n"
                "    del s\n"
                "    import gc\n"
                "    gc.collect()\n"
            ) % remote
            out = r.raw_exec(code, timeout=timeout)
            r.raw_leave()
            parts = out.replace(b"OK", b"", 1).split(b"\x04")
            stdout = parts[0] if parts else b""
            stderr = parts[1] if len(parts) > 1 else b""
            print(stdout.decode("utf-8", "replace"))
            if stderr.strip():
                print("---------- STDERR ----------")
                print(stderr.decode("utf-8", "replace"))
        elif cmd == "reset":
            # 硬复位：清掉残留 sensor/display 与泄漏的堆，等待 banner 出现
            print("复位板子 ...")
            r.friendly()
            r.exec_line("import machine; machine.reset()", wait=0.3)
            time.sleep(1.2)
            buf = b""
            t0 = time.time()
            while time.time() - t0 < 15:
                n = r.ser.in_waiting
                if n:
                    buf += r.ser.read(n)
                    if b"CanMV" in buf and b">>>" in buf:
                        break
                else:
                    time.sleep(0.2)
            tail = buf.decode("utf-8", "replace").strip().splitlines()
            print("\n".join(tail[-4:]) if tail else "(无 banner)")
            r.friendly()
        elif cmd == "get":
            remote, local = args[1], args[2]
            t0 = time.time()
            data = r.read_file(remote, timeout=15.0, verbose=True)
            el = time.time() - t0
            d = __import__("os").path.dirname(local)
            if d:
                __import__("os").makedirs(d, exist_ok=True)
            open(local, "wb").write(data)
            print(
                "\n下载 %s -> %s  %d 字节  %.1f s  %.1f KB/s"
                % (remote, local, len(data), el, len(data) / 1024.0 / el if el else 0)
            )
        elif cmd == "cmd":
            show("执行", r.exec_block(args[1]))
        else:
            print("未知命令:", cmd)
    finally:
        r.close()


if __name__ == "__main__":
    main()
