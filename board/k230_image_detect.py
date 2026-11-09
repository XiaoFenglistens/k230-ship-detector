# SPDX-FileCopyrightText: 2026 XiaoFenglistens
# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""K230 影像文件检测（不用摄像头）

把遥感影像放进 TF 卡 /sdcard/input/，本脚本逐张检测并输出标注图到 /sdcard/photo/。
链路：JPEG 解码 -> RGB888 -> numpy(H,W,3) -> ai2d(RGB_packed, pad+resize) -> KPU -> 后处理 -> 画框 -> 存 RGB565

用法：
  1. 建 /sdcard/input/ 放进 .jpg
  2. 运行本脚本，或把它改名为 main.py 开机跑
"""
import os
import gc
import time

import image
import ulab.numpy as np
import nncase_runtime as nn
import aidemo
from libs.AI2D import Ai2d
from media.display import *
from media.media import *

KMODEL = "/sdcard/best.kmodel"
IN_DIRS = ["/sdcard/input", "/sdcard"]
OUT_DIR = "/sdcard/photo"
MODEL_IN = [416, 416]
CONF_THRESH = 0.30
NMS_THRESH = 0.45
MAX_BOXES = 200
LABELS = ["ship"]
LABEL_LIMIT = 25          # 超过这么多目标就只画框不画字
JPEG_QUALITY = 92

kpu = None
DISPLAY_OK = False


def ensure_dir(p):
    try:
        os.stat(p)
    except OSError:
        try:
            os.mkdir(p)
        except OSError:
            pass


def show_status(lines, title="IMAGE DETECT"):
    """在 LCD 上显示一张状态卡片（结果图比屏幕大时用）。"""
    if not DISPLAY_OK:
        return
    try:
        card = image.Image(800, 480, image.RGB565)
        card.clear()
        card.draw_string_advanced(28, 24, 32, title, color=(0, 255, 0))
        y = 88
        for ln in lines:
            card.draw_string_advanced(28, y, 24, ln, color=(255, 255, 255))
            y += 40
        Display.show_image(card)
        card = None
        gc.collect()
    except Exception as e:
        print("   (status show failed: %s)" % e)


def write_json(path, W, H, dets, pad_x, pad_y):
    """把检测结果写成 JSON（坐标已还原到原图，便于电脑端使用）。"""
    n = len(dets[0])
    parts = []
    for i in range(n):
        x, y, w, h = dets[0][i]
        x = int(x) - pad_x
        y = int(y) - pad_y
        parts.append('{"cls": %d, "conf": %.3f, "x": %d, "y": %d, "w": %d, "h": %d}'
                     % (int(dets[1][i]), float(dets[2][i]), x, y, int(w), int(h)))
    body = '{"image_w": %d, "image_h": %d, "count": %d, "detections": [%s]}\n' % (
        W, H, n, ", ".join(parts))
    f = open(path, "w")
    f.write(body)
    f.close()
    return n


def draw_dets(img, dets, pad_x, pad_y):
    n = len(dets[0])
    dense = n > LABEL_LIMIT
    for i in range(n):
        x, y, w, h = dets[0][i]
        x = int(x) - pad_x
        y = int(y) - pad_y
        w = int(w)
        h = int(h)
        img.draw_rectangle(x, y, w, h, color=(255, 60, 60), thickness=3)
        if dense:
            continue
        ty = y - 32
        if ty < 0:
            ty = y + 3
        img.draw_string_advanced(x, ty, 26,
                                 "%s %.2f" % (LABELS[int(dets[1][i])], dets[2][i]),
                                 color=(255, 255, 0))
    return n


def detect_image(path, out_path, show=False):
    """返回 (目标数, 耗时毫秒)。同时写出同名 .json 检测结果。"""
    t0 = time.ticks_ms()
    src = image.Image(path)
    W, H = src.width(), src.height()
    rgb = src.to_rgb888()
    src = None
    arr = rgb.to_numpy_ref()

    # 按比例缩放到"内框"，再在 numpy 里补灰边到 MODEL_IN。
    # 注意：ai2d.pad() 在 RGB_packed(RGB888) 布局下不生效（实测：640 宽图
    # 被直接拉成正方形，纵向放大 1.67 倍），所以 padding 只能放 numpy 做。
    sc = MODEL_IN[0] * 1.0 / max(W, H)
    nw = int(W * sc + 0.5)
    nh = int(H * sc + 0.5)
    if nw < 2:
        nw = 2
    if nh < 2:
        nh = 2
    left = (MODEL_IN[0] - nw) // 2
    top = (MODEL_IN[1] - nh) // 2

    ai = Ai2d(debug_mode=0)
    ai.set_ai2d_dtype(nn.ai2d_format.RGB_packed, nn.ai2d_format.NCHW_FMT,
                      np.uint8, np.uint8)
    ai.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
    ai.build([1, H, W, 3], [1, 3, nh, nw])
    small = ai.run(arr).to_numpy()
    ai = None
    arr = None
    gc.collect()

    canvas = np.zeros((1, 3, MODEL_IN[1], MODEL_IN[0]), dtype=np.uint8)
    if top > 0:
        canvas[0, :, 0:top, :] = 128
    if top + nh < MODEL_IN[1]:
        canvas[0, :, top + nh:, :] = 128
    if left > 0:
        canvas[0, :, :, 0:left] = 128
    if left + nw < MODEL_IN[0]:
        canvas[0, :, :, left + nw:] = 128
    canvas[0, :, top:top + nh, left:left + nw] = small
    small = None
    gc.collect()

    kpu.set_input_tensor(0, nn.from_numpy(canvas))
    kpu.run()
    out = kpu.get_output_tensor(0).to_numpy()
    new_result = out[0].transpose()
    dets = aidemo.yolov8_det_postprocess(
        new_result.copy(),
        [MODEL_IN[0], MODEL_IN[1]], [MODEL_IN[1], MODEL_IN[0]],
        [MODEL_IN[0], MODEL_IN[1]],
        len(LABELS), CONF_THRESH, NMS_THRESH, MAX_BOXES)
    canvas = None
    out = None
    gc.collect()

    # 后处理给的是 MODEL_IN 画布坐标，换算回原图像素（缩放 + 去 pad）
    sx = W * 1.0 / nw
    sy = H * 1.0 / nh
    boxes = []
    for i in range(len(dets[0])):
        x, y, w, h = dets[0][i]
        boxes.append(((x - left) * sx, (y - top) * sy, w * sx, h * sy))
    dets[0] = boxes
    left = 0
    top = 0

    n = draw_dets(rgb, dets, left, top)
    img565 = rgb.to_rgb565()
    rgb = None
    arr = None
    gc.collect()
    # 标一下原图尺寸与目标数
    img565.draw_string_advanced(6, 6, 24, "%dx%d  %d ships" % (W, H, n),
                                color=(0, 255, 0))
    img565.save(out_path, quality=JPEG_QUALITY)
    write_json(out_path[:-4] + ".json", W, H, dets, left, top)
    if show:
        base = out_path.split("/")[-1]
        if W <= 800 and H <= 480:
            try:
                Display.show_image(img565)
            except Exception as e:
                print("   (display show failed: %s)" % e)
        else:
            # 结果图比屏幕大，改成状态卡片（想看大图请把 DET_*.jpg 拷到电脑）
            show_status(["%s  %d ships" % (base, n),
                         "%d x %d  ->  %.0f ms" % (W, H, time.ticks_diff(time.ticks_ms(), t0))])
    img565 = None
    gc.collect()
    return n, time.ticks_diff(time.ticks_ms(), t0)


def main():
    global kpu, DISPLAY_OK

    display_ok = False
    try:
        MediaManager.init()
        Display.init(Display.ST7701, width=800, height=480, osd_num=1, to_ide=True)
        display_ok = True
        DISPLAY_OK = True
        print("[IMG] display ok")
    except Exception as e:
        print("[IMG] display skipped:", e)

    ensure_dir(OUT_DIR)
    ensure_dir(IN_DIRS[0])

    kpu = nn.kpu()
    kpu.load_kmodel(KMODEL)
    print("[IMG] kmodel loaded, free=%dKB" % (gc.mem_free() // 1024))

    targets = []
    for d in IN_DIRS:
        try:
            for f in os.listdir(d):
                lf = f.lower()
                # MicroPython 的 str.endswith 不支持 tuple，只能逐个判断
                if lf.endswith(".jpg") or lf.endswith(".jpeg") or lf.endswith(".bmp"):
                    targets.append(d + "/" + f)
        except OSError:
            pass
        if targets:
            break
    targets.sort()
    print("[IMG] found %d image(s)" % len(targets))
    if not targets:
        print("[IMG] 把图片放到 /sdcard/input/ 再运行")
        return

    ok = 0
    fail = 0
    for idx, p in enumerate(targets):
        base = p.split("/")[-1].rsplit(".", 1)[0]
        outp = "%s/DET_%s.jpg" % (OUT_DIR, base)
        try:
            n, ms = detect_image(p, outp, show=display_ok)
            ok += 1
            print("[IMG] %2d/%d %-28s %3d ships  %5d ms  -> %s"
                  % (idx + 1, len(targets), p.split("/")[-1], n, ms, outp.split("/")[-1]))
        except Exception as e:
            fail += 1
            print("[IMG] %2d/%d %s FAILED: %s" % (idx + 1, len(targets), p, e))
            gc.collect()
    print("[IMG] done: %d ok, %d failed, free=%dKB" % (ok, fail, gc.mem_free() // 1024))

    try:
        if display_ok:
            Display.deinit()
        MediaManager.deinit()
    except Exception:
        pass


main()
