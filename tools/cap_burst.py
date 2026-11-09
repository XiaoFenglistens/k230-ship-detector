# SPDX-FileCopyrightText: 2026 XiaoFenglistens
# SPDX-License-Identifier: MIT
"""K230 低分辨率基准集 —— 自动连拍采集脚本（板端）

为什么不用按键直接拍:
    手指按 USR 会把板子推动, 存下来的那一帧是"推完之后"的画面。
    之前拍屏实验里那 18px 系统偏移就是这么来的。
    所以这里改成: 按一下 -> 倒计时 2 秒(手拿开) -> 程序自己连拍 20 张。

用法:
    1. 把本文件传到 /sdcard/cap_burst.py
    2. 在 CanMV IDE 里 打开并运行; 或临时改名为 /sdcard/main.py 由 boot.py 拉起
    3. 屏幕显示 "第 N 组 READY" 后, 短按 USR 开始
    4. 拍完显示 "第 N 组 完成", 移动位置后再短按, 自动进入第 N+1 组
    5. 长按 USR 3 秒退出

输出: /sdcard/photo/cap/gNN_MMM.jpg  (NN=组号, MMM=张号)
"""

import gc
import os
import time

import image
from machine import FPIOA, Pin
from media.display import *
from media.media import *
from media.sensor import *

# ---------------------------------------------------------------- 配置
W, H = 800, 480
OUT_DIR = "/sdcard/photo/cap"
PER_GROUP = 20          # 每组张数
INTERVAL_MS = 800       # 张间隔
COUNTDOWN_MS = 700      # 每声倒计时间隔
BUTTON_PIN = 64         # Lite 版 USR; 标准版是 53
LED_G_PIN = 66          # 绿灯: 正在拍
LED_B_PIN = 71          # 蓝灯: 待机
DEBOUNCE_MS = 20
HOLD_MS = 3000          # 长按退出
JPEG_Q = 92

btn = None
led_g = None
led_b = None
osd = None


# ---------------------------------------------------------------- 工具
def ensure_dir(path):
    """/sdcard/photo/cap 不存在就建, 逐级建。"""
    parts = path.strip("/").split("/")
    cur = ""
    for p in parts:
        cur = cur + "/" + p
        try:
            os.stat(cur)
        except OSError:
            try:
                os.mkdir(cur)
            except OSError as e:
                print("[CAP] mkdir fail", cur, e)


def next_group():
    """扫描已有文件, 返回下一个可用的组号 (断电/重跑也不会覆盖)。"""
    mx = 0
    try:
        for n in os.listdir(OUT_DIR):
            if len(n) < 5 or n[0] != "g":
                continue
            try:
                g = int(n[1:3])
            except Exception:
                continue
            if g > mx:
                mx = g
    except OSError:
        pass
    return mx + 1


def status(lines, color=(0, 255, 255)):
    """在屏上画状态文字 (中文字库内置, 直接写汉字)。"""
    global osd
    try:
        osd.clear()
        y = 10
        for ln in lines:
            osd.draw_string_advanced(10, y, 30, ln, color=color)
            y += 44
        Display.show_image(osd, 0, 0, Display.LAYER_OSD3)
    except Exception as e:
        print("[CAP] status fail", e)


def setup_gpio():
    global btn, led_g, led_b
    fpioa = FPIOA()
    fpioa.set_function(BUTTON_PIN, FPIOA.GPIO64 if BUTTON_PIN == 64 else FPIOA.GPIO53)
    btn = Pin(BUTTON_PIN, Pin.IN, Pin.PULL_DOWN)
    fpioa.set_function(LED_G_PIN, FPIOA.GPIO66)
    fpioa.set_function(LED_B_PIN, FPIOA.GPIO71)
    led_g = Pin(LED_G_PIN, Pin.OUT, Pin.PULL_DOWN)
    led_b = Pin(LED_B_PIN, Pin.OUT, Pin.PULL_DOWN)
    led_g.value(0)
    led_b.value(0)


def wait_press(timeout_ms=None):
    """等一次按键。返回 'short' / 'long' / None(超时)。"""
    t0 = time.ticks_ms()
    pressed_at = None
    while True:
        now = time.ticks_ms()
        try:
            v = btn.value()
        except Exception:
            v = 0
        if v == 1 and pressed_at is None:
            pressed_at = now
        elif v == 1 and pressed_at is not None:
            if time.ticks_diff(now, pressed_at) >= HOLD_MS:
                return "long"
        elif v == 0 and pressed_at is not None:
            held = time.ticks_diff(now, pressed_at)
            pressed_at = None
            if held >= DEBOUNCE_MS:
                return "short"
        if timeout_ms is not None and time.ticks_diff(now, t0) > timeout_ms:
            return None
        time.sleep_ms(10)


def capture_group(sensor, group):
    """倒计时后连拍 PER_GROUP 张。"""
    for i in (3, 2, 1):
        status(["第 %d 组 准备" % group, "倒计时 %d  手请拿开" % i], (255, 255, 0))
        time.sleep_ms(COUNTDOWN_MS)
    led_b.value(0)
    led_g.value(1)

    ok = 0
    t0 = time.ticks_ms()
    for i in range(PER_GROUP):
        try:
            img = sensor.snapshot(chn=CAM_CHN_ID_1)
            path = "%s/g%02d_%03d.jpg" % (OUT_DIR, group, i + 1)
            img.save(path, quality=JPEG_Q)
            ok += 1
        except Exception as e:
            print("[CAP] save fail", i, e)
        if (i + 1) % 5 == 0:
            status(["第 %d 组" % group, "已拍 %d / %d" % (i + 1, PER_GROUP)], (0, 255, 0))
        time.sleep_ms(INTERVAL_MS)

    led_g.value(0)
    led_b.value(1)
    dt = time.ticks_diff(time.ticks_ms(), t0) / 1000.0
    status(["第 %d 组 完成" % group, "%d 张 / %.1f 秒" % (ok, dt), "按 USR 继续, 长按退出"], (0, 255, 0))
    print("[CAP] group %d done, %d shots in %.1f s" % (group, ok, dt))
    return ok


# ---------------------------------------------------------------- 主流程
def main():
    global osd

    # 兜底释放上一轮的资源
    for fn in (lambda: MediaManager.deinit(), lambda: Sensor.deinit(), lambda: Display.deinit()):
        try:
            fn()
        except Exception:
            pass

    ensure_dir(OUT_DIR)
    setup_gpio()

    sensor = Sensor(fps=30)
    sensor.reset()
    # 三通道必须在 run() 之前配完
    sensor.set_framesize(W, H, chn=CAM_CHN_ID_0)
    sensor.set_pixformat(Sensor.YUV420SP, chn=CAM_CHN_ID_0)   # 只给显示
    sensor.set_framesize(W, H, chn=CAM_CHN_ID_1)
    sensor.set_pixformat(Sensor.RGB565, chn=CAM_CHN_ID_1)     # 存图(唯一可 save 的格式)
    Display.init(Display.ST7701, width=W, height=H, osd_num=1, to_ide=True)
    MediaManager.init()
    osd = image.Image(W, H, image.ARGB8888)
    Display.bind_layer(**sensor.bind_info(x=0, y=0, chn=CAM_CHN_ID_0), layer=Display.LAYER_VIDEO1)
    sensor.run()

    group = next_group()
    led_b.value(1)
    status([
        "低分辨率基准集 采集",
        "下一组: 第 %d 组" % group,
        "短按 USR 开始   长按退出",
    ])
    print("[CAP] ready, next group =", group)

    try:
        while True:
            ev = wait_press()
            if ev == "long":
                print("[CAP] long press, exit")
                break
            if ev != "short":
                continue
            capture_group(sensor, group)
            group += 1
            gc.collect()
            status([
                "第 %d 组 完成" % (group - 1),
                "下一组: 第 %d 组" % group,
                "短按 USR 继续   长按退出",
            ], (0, 255, 0))
    except KeyboardInterrupt:
        print("[CAP] ctrl-c")
    except Exception as e:
        print("[CAP] EXCEPTION:", e)
        try:
            status(["出错", str(e)[:40]], (255, 0, 0))
            time.sleep_ms(1500)
        except Exception:
            pass
        raise
    finally:
        try:
            led_g.value(0)
            led_b.value(0)
        except Exception:
            pass
        for fn in (lambda: sensor.stop(), lambda: Display.deinit(), lambda: MediaManager.deinit()):
            try:
                fn()
            except Exception as e:
                print("[CAP] cleanup", e)
        gc.collect()
        print("[CAP] done")


main()
