# -*- coding: utf-8 -*-
"""
K230D 庐山派 Lite —— 船舶检测 + USR 按键拍照存卡

硬件前提：
  立创·庐山派 Lite-K230D-CanMV / 3.1 寸 MIPI 屏 800x480 / TF 卡
用法：
  1. 把 best.kmodel 放到 TF 卡根目录
  2. 本文件改名 main.py 放 TF 卡根目录（或手动运行）
  3. 上电后屏幕显示摄像头画面 + 实时船舶检测框
  4. 短按 USR 键 -> 拍一张照片存到 /sdcard/photo/IMG_xxxx.jpg
  5. 长按 USR 键 3 秒 -> 退出程序

设计要点（踩过的坑，勿改）：
  * 显示必须用 st7701（LCD），不能用 hdmi
  * sensor 需要三个通道：
      CH0 YUV420SP -> 直接绑到显示层，画面不进 CPU
      CH1 RGB565   -> 专门拍照。Image.save() 只支持 RGB565，
                      YUV420SP / RGBP888 / RGB888 都会报
                      "current format not support save function!"
      CH2 RGBP888  -> planar CHW，喂给 KPU 做推理
  * YOLOv8 必须显式传 display_size=[800,480]，否则默认 [1920,1080]，
    后处理会把框坐标按 1080p 缩放，框全部飞出画面
  * YOLOv8 创建后必须显式调 config_preprocess()，固件里没有自动调用，
    否则 ai2d_builder 为 None，推理时报 'NoneType' object has no attribute 'run'
  * max_boxes_num 默认 50，密集港口图会被截断，这里设 200
"""
import os
import gc
import time

import image
import ujson
from media.sensor import *
from media.display import *
from media.media import *
from machine import Pin, FPIOA
from libs.YOLO import YOLOv8

# ---------------- 可调参数 ----------------
KMODEL_PATH = "/sdcard/best640.kmodel"
LABELS = ["ship"]
MODEL_INPUT_SIZE = [640, 640]
RGB888P_SIZE = [800, 480]          # 与屏幕一致：AI 坐标系 = 显示坐标系 = 照片坐标系
DISPLAY_SIZE = [800, 480]
DISPLAY_MODE = "st7701"            # LCD，不是 hdmi
CONFIDENCE_THRESHOLD = 0.30
NMS_THRESHOLD = 0.45
MAX_BOXES_NUM = 200

BUTTON_PIN = 64                    # Lite 版 USR = GPIO64（非 Lite 版是 GPIO53）
LED_R_PIN = 65
LED_G_PIN = 66
KEY_PRESSED = 1
DEBOUNCE_MS = 20
HOLD_MS = 3000

SAVE_DIRS = ["/sdcard/photo", "/sdcard", "/data"]
JPEG_QUALITY = 92
GC_INTERVAL = 30

# 目标数超过 LABEL_LIMIT 时只画框不画文字，避免标签糊满屏
LABEL_LIMIT = 25


# ---------------- 小工具 ----------------
def pick_save_dir():
    """找一个可写的目录，返回路径与已用到的最大序号。"""
    for d in SAVE_DIRS:
        try:
            os.stat(d)
        except OSError:
            try:
                os.mkdir(d)
            except OSError:
                continue
        try:
            names = os.listdir(d)
        except OSError:
            names = []
        seq = 0
        for n in names:
            if n.startswith("IMG_") and n.endswith(".jpg") and len(n) == 12:
                try:
                    v = int(n[4:8])
                    if v > seq:
                        seq = v
                except ValueError:
                    pass
        return d, seq
    return "/data", 0


def setup_gpio():
    """USR 按键（下拉、按下为高）+ RGB 灯。

    注意：这个固件里 FPIOA 是**类**，必须实例化后再 set_function；
    不存在 machine.fpioa 模块（写 from machine import fpioa 会 ImportError）。
    """
    fpioa = FPIOA()
    fpioa.set_function(BUTTON_PIN, FPIOA.GPIO64)
    key = Pin(BUTTON_PIN, Pin.IN, Pin.PULL_DOWN, drive=7)

    fpioa.set_function(LED_R_PIN, FPIOA.GPIO65)
    fpioa.set_function(LED_G_PIN, FPIOA.GPIO66)
    led_r = Pin(LED_R_PIN, Pin.OUT, drive=7)
    led_g = Pin(LED_G_PIN, Pin.OUT, drive=7)
    led_r.value(0)
    led_g.value(0)
    return key, led_r, led_g


def draw_boxes(img, res, thickness=3, text_size=26):
    """把检测结果画到任意 Image 上（不清理背景，用于照片叠加）。

    目标很多时不再逐个画标签，否则几十个 "ship 0.xx" 会糊成一片看不清画面。
    规则：总数 <= LABEL_LIMIT 时全画标签；超过则只画框，不再画文字
    （总数已经显示在左下角，密排场景保留画面比保留文字更有用）。
    """
    n = len(res[0])
    dense = n > LABEL_LIMIT
    for i in range(n):
        x, y, w, h = res[0][i]
        conf = res[2][i]
        img.draw_rectangle(int(x), int(y), int(w), int(h),
                           color=(255, 60, 60), thickness=thickness)
        if dense:
            continue
        # 标签放在框上方，贴近顶部时改放框内
        ty = int(y) - text_size - 6
        if ty < 0:
            ty = int(y) + 4
        img.draw_string_advanced(int(x), ty, text_size,
                                 "%s %.2f" % (LABELS[int(res[1][i])], conf),
                                 color=(255, 255, 0))
    return n


def write_det_json(jpath, seq, dets, fps_val, w, h):
    """把这一帧的检测结果写成 JSON（逐条写，避免一次性拼出大字符串）。"""
    n = len(dets[0])
    f = open(jpath, "w")
    try:
        f.write('{"seq": %d, "uptime_ms": %d, "image_w": %d, "image_h": %d, "fps": %.1f, "count": %d, "detections": ['
                % (seq, time.ticks_ms(), w, h, fps_val, n))
        for i in range(n):
            x, y, bw, bh = dets[0][i]
            f.write('%s{"cls": %d, "conf": %.3f, "x": %d, "y": %d, "w": %d, "h": %d}'
                    % ("" if i == 0 else ", ", int(dets[1][i]), float(dets[2][i]),
                       int(x), int(y), int(bw), int(bh)))
        f.write("]}\n")
    finally:
        f.close()
    return n


def append_shot_log(csvpath, seq, fps_val, n, fname):
    """往 photos.csv 追加一行汇总，方便在电脑上一眼扫完所有照片。"""
    exists = True
    try:
        os.stat(csvpath)
    except OSError:
        exists = False
    f = open(csvpath, "a")
    try:
        if not exists:
            f.write("seq,uptime_ms,fps,count,file\n")
        f.write("%d,%d,%.1f,%d,%s\n" % (seq, time.ticks_ms(), fps_val, n, fname))
    finally:
        f.close()


def main():
    key = None
    led_r = None
    led_g = None
    sensor = None
    yolo = None

    try:
        # ---------- 1. 清掉上电可能残留的资源 ----------
        for fn in (MediaManager.deinit, Sensor.deinit, Display.deinit):
            try:
                fn()
            except Exception:
                pass
        gc.collect()

        # ---------- 2. GPIO ----------
        key, led_r, led_g = setup_gpio()
        led_r.value(1)
        print("[DET] GPIO ok, USR=GPIO%d" % BUTTON_PIN)

        # ---------- 3. sensor 三通道 ----------
        sensor = Sensor(fps=30)
        sensor.reset()
        # CH0：给显示，YUV420SP
        sensor.set_framesize(w=DISPLAY_SIZE[0], h=DISPLAY_SIZE[1], chn=CAM_CHN_ID_0)
        sensor.set_pixformat(Sensor.YUV420SP, chn=CAM_CHN_ID_0)
        # CH1：给拍照，RGB565（唯一能 save 的格式）
        sensor.set_framesize(w=DISPLAY_SIZE[0], h=DISPLAY_SIZE[1], chn=CAM_CHN_ID_1)
        sensor.set_pixformat(Sensor.RGB565, chn=CAM_CHN_ID_1)
        # CH2：给 AI，RGBP888（planar CHW）
        sensor.set_framesize(w=RGB888P_SIZE[0], h=RGB888P_SIZE[1], chn=CAM_CHN_ID_2)
        sensor.set_pixformat(Sensor.RGBP888, chn=CAM_CHN_ID_2)
        print("[DET] sensor configured (CH0 YUV420SP / CH1 RGB565 / CH2 RGBP888)")

        # ---------- 4. 显示 ----------
        display_type = Display.ST7701 if DISPLAY_MODE == "st7701" else Display.ST7701
        Display.init(display_type, width=DISPLAY_SIZE[0], height=DISPLAY_SIZE[1],
                     osd_num=1, to_ide=True)
        osd = image.Image(DISPLAY_SIZE[0], DISPLAY_SIZE[1], image.ARGB8888)
        Display.bind_layer(**sensor.bind_info(x=0, y=0, chn=CAM_CHN_ID_0),
                           layer=Display.LAYER_VIDEO1)
        sensor.run()
        print("[DET] display %dx%d, sensor running" % (Display.width(), Display.height()))

        # ---------- 5. 加载模型 ----------
        try:
            os.stat(KMODEL_PATH)
        except OSError:
            Display.show_image(osd, 0, 0, Display.LAYER_OSD3)
            osd.clear()
            osd.draw_string_advanced(20, 40, 28, "KMODEL NOT FOUND", color=(255, 0, 0))
            osd.draw_string_advanced(20, 90, 22, KMODEL_PATH, color=(255, 255, 255))
            Display.show_image(osd, 0, 0, Display.LAYER_OSD3)
            raise OSError("kmodel missing: " + KMODEL_PATH)

        yolo = YOLOv8(task_type="detect", mode="video", kmodel_path=KMODEL_PATH,
                      labels=LABELS,
                      rgb888p_size=RGB888P_SIZE,
                      model_input_size=MODEL_INPUT_SIZE,
                      display_size=DISPLAY_SIZE,
                      conf_thresh=CONFIDENCE_THRESHOLD,
                      nms_thresh=NMS_THRESHOLD,
                      max_boxes_num=MAX_BOXES_NUM,
                      debug_mode=0)
        yolo.config_preprocess()
        print("[DET] kmodel loaded")
        led_r.value(0)
        led_g.value(1)

        save_dir, seq = pick_save_dir()
        print("[DET] save dir %s (next #%d)" % (save_dir, seq + 1))

        # ---------- 6. 主循环 ----------
        frame_id = 0
        fps_t0 = time.ticks_ms()
        fps_frames = 0
        fps_val = 0.0
        last_raw = 0
        last_tick = time.ticks_ms()
        hold_start = None
        long_fired = False
        det_n = 0
        # 存图提示：保留 2.5 秒再消失（否则下一帧 draw_result 的 clear 会立刻抹掉）
        msg_text = ""
        msg_color = (0, 255, 255)
        msg_until = 0

        while True:
            # 6.1 取 AI 帧 + 推理
            ai_img = sensor.snapshot(chn=CAM_CHN_ID_2)
            res = yolo.run(ai_img.to_numpy_ref())
            det_n = len(res[0]) if res else 0

            # 6.2 画到 OSD 并显示（用自家的 draw_boxes，标签策略与存图一致）
            osd.clear()
            draw_boxes(osd, res, thickness=2, text_size=24)
            fps_frames += 1
            now = time.ticks_ms()
            if time.ticks_diff(now, fps_t0) >= 1000:
                fps_val = fps_frames * 1000.0 / time.ticks_diff(now, fps_t0)
                fps_t0 = now
                fps_frames = 0
            osd.draw_string_advanced(6, 6, 30,
                                     "%.1f fps  %d ships" % (fps_val, det_n),
                                     color=(0, 255, 0))
            if time.ticks_diff(msg_until, now) > 0:
                osd.draw_string_advanced(6, 44, 28, msg_text, color=msg_color)
            Display.show_image(osd, 0, 0, Display.LAYER_OSD3)

            # 6.3 按键状态机
            raw = key.value()
            if raw != last_raw:
                last_tick = now
                last_raw = raw
            if time.ticks_diff(now, last_tick) >= DEBOUNCE_MS:
                if raw == KEY_PRESSED:
                    if hold_start is None:
                        hold_start = now
                        long_fired = False
                    elif (not long_fired) and time.ticks_diff(now, hold_start) >= HOLD_MS:
                        long_fired = True
                        print("[DET] long press -> exit")
                        break
                else:
                    if hold_start is not None:
                        held = time.ticks_diff(now, hold_start)
                        hold_start = None
                        if (not long_fired) and held >= 60:
                            # 短按：拍一张
                            led_r.value(1)
                            try:
                                seq += 1
                                fname = "%s/IMG_%04d.jpg" % (save_dir, seq)
                                photo = sensor.snapshot(chn=CAM_CHN_ID_1)   # RGB565
                                n = draw_boxes(photo, res)
                                photo.draw_string_advanced(6, DISPLAY_SIZE[1] - 34, 24,
                                                           "%d ships  %.1f fps" % (n, fps_val),
                                                           color=(0, 255, 0))
                                photo.save(fname, quality=JPEG_QUALITY)
                                photo = None
                                gc.collect()

                                # 结果数据一并落盘：同名 .json（逐框坐标）+ photos.csv（一行汇总）
                                short = fname.split("/")[-1]
                                try:
                                    write_det_json(fname[:-4] + ".json", seq, res,
                                                   fps_val, DISPLAY_SIZE[0], DISPLAY_SIZE[1])
                                    append_shot_log(save_dir + "/photos.csv", seq, fps_val, n, short)
                                except Exception as e2:
                                    print("[DET] json log failed:", e2)

                                print("[DET] saved %s (%d ships)" % (fname, n))
                                msg_text = "SAVED %s  %d" % (short, n)
                                msg_color = (0, 255, 255)
                                msg_until = time.ticks_add(time.ticks_ms(), 2500)
                            except Exception as e:
                                print("[DET] SAVE FAILED:", e)
                                msg_text = "SAVE FAILED"
                                msg_color = (255, 60, 60)
                                msg_until = time.ticks_add(time.ticks_ms(), 2500)
                            finally:
                                led_r.value(0)

            # 6.4 定期回收
            frame_id += 1
            if frame_id % GC_INTERVAL == 0:
                gc.collect()
                ai_img = None

    except KeyboardInterrupt:
        print("[DET] interrupted")
    except Exception as e:
        print("[DET] exception:", e)
        try:
            if osd is not None:
                osd.clear()
                osd.draw_string_advanced(20, 40, 26, "ERROR", color=(255, 0, 0))
                osd.draw_string_advanced(20, 90, 20, str(e)[:60], color=(255, 255, 255))
                Display.show_image(osd, 0, 0, Display.LAYER_OSD3)
        except Exception:
            pass
        raise
    finally:
        # ---------- 7. 收尾 ----------
        try:
            if yolo is not None:
                yolo.deinit()
        except Exception:
            pass
        try:
            if sensor is not None:
                sensor.stop()
        except Exception:
            pass
        try:
            Display.deinit()
        except Exception:
            pass
        try:
            MediaManager.deinit()
        except Exception:
            pass
        try:
            if led_r is not None:
                led_r.value(0)
            if led_g is not None:
                led_g.value(0)
        except Exception:
            pass
        gc.collect()
        print("[DET] clean exit")


if __name__ == "__main__":
    main()
