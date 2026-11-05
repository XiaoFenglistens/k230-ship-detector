# SPDX-FileCopyrightText: 2026 XiaoFenglistens
# SPDX-License-Identifier: MIT
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8 ONNX -> K230 kmodel 转换脚本（nncase 2.11.0）

参数对齐 DFRobot/onnx2kmodel 的 kmodel_conf.toml：
  preprocess=true  input_type=uint8  input_range=[0,1]
  mean=[0,0,0]  std=[1,1,1]  swapRB=false  letterbox_value=0
  quant_type=uint8（权重与激活均 uint8）

关键点：
  * 校准数据必须与训练/推理时的通道顺序一致 —— ultralytics 用 RGB，
    故 cv2.imread 得到 BGR 要 cvtColor 成 RGB，否则颜色错乱。
  * letterbox 用 0 填充（黑边），与 ultralytics 默认 (114,114,114) 不同；
    由于本数据集图片均为 800x800 正方形，padding 恒为 0，二者等价。
  * ptq.set_tensor_data 传入的是 [N,1,3,H,W] 的 list（外层再套一个 list）。

用法：
  # 默认：NoClip / 60 张校准 / 输出 weights\\best.kmodel
  python tools\\convert_kmodel.py

  # 换校准算法（对比量化损失时用）
  python tools\\convert_kmodel.py --calibrate KLD --out ...\\best_kld.kmodel
  python tools\\convert_kmodel.py --calibrate KLD --nsamp 120 --out ...\\best_kld120.kmodel
"""
import os as _os
KROOT = _os.environ.get("K230_ROOT", ".")  # 项目根目录，默认当前目录

import os
import sys
import glob
import time

# ---- nncase 插件路径必须在 import nncase 之前设好 ----
import importlib.util

_spec = importlib.util.find_spec("nncase")
_pkg_dir = os.path.dirname(os.path.abspath(_spec.origin))
_site = os.path.dirname(_pkg_dir)
os.environ["NNCASE_PLUGIN_PATH"] = _site
if _site not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _site + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import cv2
import nncase
import nncase_kpu  # noqa: F401  注册 k230 target


def _arg(name, default):
    """极简参数：-name value"""
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


RUN_DIR = _arg("--run", _os.path.join(KROOT, 'runs', 'ship_416_v1'))
ONNX = os.path.join(RUN_DIR, "weights", "best.onnx")
OUT = _arg("--out", os.path.join(RUN_DIR, "weights", "best.kmodel"))
CALIB = _arg("--calib", _os.path.join(KROOT, 'calib'))
SIZE = int(_arg("--size", "416"))
NSAMP = int(_arg("--nsamp", "60"))
CALIBRATE = _arg("--calibrate", "NoClip")     # NoClip / KLD / EasyQuant
W_QUANT = _arg("--w-quant", "uint8")          # uint8 / int8
A_QUANT = _arg("--a-quant", "uint8")


def preprocess(path, size):
    """BGR 文件 -> RGB uint8 CHW（letterbox 到正方形）"""
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError("read failed: %s" % path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = max(w, h)
    pt, pl = (scale - h) // 2, (scale - w) // 2
    pb, pr = (scale - h) - pt, (scale - w) - pl
    if pt or pb or pl or pr:
        img = cv2.copyMakeBorder(img, pt, pb, pl, pr,
                                 cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    return np.transpose(img, (2, 0, 1))


def main():
    files = sorted(glob.glob(os.path.join(CALIB, "*.jpg")))[:NSAMP]
    print("[1/4] 校准图 %d 张（calibrate=%s）" % (len(files), CALIBRATE))
    data = np.stack([preprocess(f, SIZE) for f in files]).astype(np.uint8)
    data = np.expand_dims(data, axis=1)          # (N, 1, 3, H, W)
    print("      校准张量 shape =", data.shape,
          " dtype =", data.dtype, " MB = %.1f" % (data.nbytes / 1048576))

    co = nncase.CompileOptions()
    co.target = "k230"
    co.dump_ir = False
    co.dump_asm = False
    co.dump_dir = "./dump"
    co.input_file = ""
    co.preprocess = True
    co.input_type = "uint8"
    co.input_shape = [1, 3, SIZE, SIZE]
    co.input_range = [0, 1]
    co.input_layout = "NCHW"
    co.swapRB = False
    co.mean = [0, 0, 0]
    co.std = [1, 1, 1]
    co.letterbox_value = 0
    co.output_layout = "NCHW"

    compiler = nncase.Compiler(co)
    print("[2/4] 导入 ONNX ...")
    with open(ONNX, "rb") as f:
        compiler.import_onnx(f.read(), nncase.ImportOptions())
    print("      import ok")

    ptq = nncase.PTQTensorOptions()
    ptq.calibrate_method = CALIBRATE
    ptq.finetune_weights_method = "NoFineTuneWeights"
    ptq.quant_type = A_QUANT
    ptq.w_quant_type = W_QUANT
    ptq.quant_scheme = ""
    ptq.quant_scheme_strict_mode = False
    ptq.export_quant_scheme = False
    ptq.export_weight_range_by_channel = False
    ptq.dump_quant_error = False
    ptq.dump_quant_error_symmetric_for_signed = False
    ptq.samples_count = data.shape[0]
    ptq.set_tensor_data([data])
    compiler.use_ptq(ptq)

    print("[3/4] 编译（PTQ + codegen）...")
    t0 = time.time()
    compiler.compile()
    print("      编译完成 %.1f s" % (time.time() - t0))

    print("[4/4] 写出 kmodel ...")
    blob = compiler.gencode_tobytes()
    with open(OUT, "wb") as f:
        f.write(blob)
    print("OK %s  %.2f MB" % (OUT, os.path.getsize(OUT) / 1048576))


if __name__ == "__main__":
    sys.exit(main())
