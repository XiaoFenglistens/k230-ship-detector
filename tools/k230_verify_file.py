# SPDX-FileCopyrightText: 2026 XiaoFenglistens
# SPDX-License-Identifier: MIT
"""比对板端文件与本地文件（大小 + 采样点字节）。

用法：python tools\\k230_verify_file.py <板端路径> <本地路径>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from k230_repl import Repl  # noqa: E402


def main():
    remote, local = sys.argv[1], sys.argv[2]
    size = os.path.getsize(local)
    offsets = [0, size // 3, (2 * size) // 3, size - 32]

    r = Repl()
    r.raw_enter()
    code = (
        "import os\n"
        "p=%r\n"
        "print('SIZE', os.stat(p)[6])\n"
        "f=open(p,'rb')\n"
        "for off in %r:\n"
        "    f.seek(off)\n"
        "    print('CHUNK', off, f.read(32).hex())\n"
        "f.close()\n" % (remote, offsets)
    )
    out = r.raw_exec(code, timeout=90.0).decode("utf-8", "replace")
    r.raw_leave()
    r.close()

    remote_size = None
    remote_chunks = {}
    for line in out.splitlines():
        if line.startswith("SIZE "):
            remote_size = int(line.split()[1])
        elif line.startswith("CHUNK "):
            _, off, hx = line.split()
            remote_chunks[int(off)] = hx
    print("板端大小 =", remote_size, " 本地大小 =", size)

    ok = remote_size == size
    with open(local, "rb") as f:
        for off in offsets:
            f.seek(off)
            want = f.read(32).hex()
            got = remote_chunks.get(off, "")
            same = want == got
            ok = ok and same
            print("  偏移 %-9d %s  %s" % (off, "一致" if same else "不一致",
                                         "" if same else "本地 %s / 板端 %s" % (want, got)))
    print("总体:", "全部一致 ✔" if ok else "有不一致 ✘")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
