# -*- coding: utf-8 -*-
"""K230 触摸启动器 v2 —— 20 类遥感目标检测

开机动画 -> 主页（5 项分级菜单）-> 类别选择 -> 实时检测
设计要点：
  * 显示用 ARGB OSD + CH0 视频层（必须 MediaManager.init()，否则黑屏）
  * 屏上文字仅 ASCII（固件无中文字库）
  * 触摸：裸 I2C 读 FT5x06（0x38），四角标定映射
  * 推理拆三步，中间 gc，防 20 类后处理爆内存
"""

import os
import gc
import time
import image
import nncase_runtime as nn
from media.sensor import *
from media.display import *
from media.media import *
from machine import Pin, FPIOA, I2C
from libs.YOLO import YOLOv8

# ---------------- 常量 ----------------
DISPLAY_SIZE = [800, 480]
RGB888P_SIZE = [800, 480]
MODEL_INPUT_SIZE = [416, 416]
KMODEL_PATH = "/sdcard/dior20_a16.kmodel"

CONF_THRESH = 0.30
NMS_THRESH = 0.45
MAX_BOXES = 100
GC_INTERVAL = 4
JPEG_QUALITY = 90

SAVE_DIRS = ["/sdcard/photo", "/sdcard", "/data"]

# ---- 边缘按钮统一往内收（贴近屏幕边缘时触摸容易点不中）----
EDGE_TOP = 12
EDGE_RIGHT = 16
EDGE_BOTTOM = 22
EDGE_LEFT = 18
EXIT_RECT = (DISPLAY_SIZE[0] - EDGE_RIGHT - 124, EDGE_TOP, 124, 46)
BACK_RECT = (DISPLAY_SIZE[0] - EDGE_RIGHT - 152, EDGE_TOP, 152, 46)
PHOTO_RECT = (EDGE_LEFT, DISPLAY_SIZE[1] - EDGE_BOTTOM - 56, 196, 56)
MENU_RECT = (DISPLAY_SIZE[0] - EDGE_RIGHT - 2 - 196, DISPLAY_SIZE[1] - EDGE_BOTTOM - 56, 196, 56)

HISTORY_PATH = "/sdcard/det_history.txt"

BUTTON_PIN = 64
LED_R_PIN = 65
LED_G_PIN = 66
KEY_PRESSED = 1
HOLD_MS = 3000
DEBOUNCE_MS = 20

TOUCH_I2C_ID = 3
TOUCH_ADDR = 0x38
TM_A = -0.01434
TM_B = 0.99843
TM_C = 13.09
TM_D = 1.02984
TM_E = 0.00679
TM_F = -12.13
TOUCH_REARM_MS = 120

LABELS = ["airplane", "airport", "baseball", "basketball", "bridge",
          "chimney", "dam", "expwy", "toll", "golf",
          "track", "harbor", "overpass", "ship", "stadium",
          "tank", "tennis", "station", "vehicle", "windmill"]
CNNAME = ["飞机", "机场", "棒球场", "篮球场", "桥",
          "烟囱", "水坝", "服务区", "收费站", "高尔夫",
          "田径场", "港口", "立交桥", "船舶", "体育场",
          "储罐", "网球场", "火车站", "车辆", "风车"]

CN = ["Plane", "Airport", "Baseball", "Basket", "Bridge",
      "Chimney", "Dam", "Service Area", "Toll Gate", "Golf",
      "Track", "Harbor", "Overpass", "Ship", "Stadium",
      "Tank", "Tennis", "Station", "Vehicle", "Windmill"]

NOTE = ["停放或飞行的飞机",
        "跑道与停机坪",
        "棒球场内场",
        "单块篮球场地",
        "跨水或跨路的长桥",
        "高烟囱，带长影子",
        "河流上的拦水坝",
        "高速公路服务区",
        "高速公路收费站",
        "高尔夫球场的草坪",
        "椭圆形跑道",
        "整片码头水域",
        "道路立体交叉",
        "单条船、游艇或帆船",
        "环形看台与场地",
        "圆形储油罐",
        "带中网的网球场",
        "火车站站房",
        "汽车、卡车或巴士",
        "田野上的风力发电机"]

COLORS = [(255, 96, 96), (255, 159, 64), (255, 214, 64), (196, 228, 64),
          (140, 220, 80), (64, 214, 140), (64, 214, 214), (80, 176, 255),
          (96, 120, 255), (150, 110, 255), (200, 100, 255), (255, 96, 220),
          (255, 112, 168), (255, 232, 64), (120, 255, 168), (96, 208, 255),
          (168, 168, 255), (255, 168, 96), (200, 255, 120), (120, 255, 240)]

NC = len(LABELS)
ALL_IDX = NC

BG = (16, 17, 24)
CARD = (32, 35, 48)
CARD_HI = (46, 52, 74)
LINE = (72, 80, 104)
WHITE = (240, 242, 248)
GREY = (150, 158, 178)
GREEN = (110, 226, 150)
YELLOW = (255, 222, 92)
RED = (255, 110, 110)
CYAN = (110, 226, 226)

# ---------------- 页面 ----------------
P_HOME = 0
P_CLASS = 1
P_DETECT = 2
P_GUIDE = 3
P_USAGE = 4
P_HISTORY = 5
P_SYSTEM = 6
P_DETAIL = 7

HOME_BTNS = [
    ("识别内容选择", "先选一个地物，再实时识别", (40, 118, 340, 96)),
    ("识别历史记录", "每次拍照都会存一条", (420, 118, 340, 96)),
    ("识别结果说明", "20 个类别分别是什么", (40, 228, 340, 96)),
    ("设备使用说明", "按键、按钮与拍摄要点", (420, 228, 340, 96)),
    ("系统与清理", "内存、存储与一键清理", (40, 338, 720, 76)),
]

ACCENTS = [(0, 229, 255), (255, 110, 199), (124, 255, 107),
           (255, 209, 102), (179, 136, 255)]


# ---------------- 工具 ----------------
DBG_PATH = "/sdcard/boot_dbg.txt"


def dbg(msg):
    try:
        f = open(DBG_PATH, "a")
        f.write("%d %s\n" % (time.ticks_ms(), msg))
        f.close()
    except Exception:
        pass


def text_w(text, size):
    n = 0.0
    for ch in text:
        o = ord(ch)
        if ch == " ":
            n += 0.34
        elif o > 0x2E80:
            n += 1.00
        else:
            n += 0.56
    return int(n * size)


def cn_of(name):
    """历史文件里存的是英文类别名，显示时转中文"""
    if name == "ALL":
        return "全部"
    try:
        return CNNAME[LABELS.index(name)]
    except Exception:
        return name


def text_at(img, x, y, text, size, color):
    img.draw_string_advanced(int(x), int(y), size, text, color=color)


def text_center(img, cx, y, text, size, color):
    text_at(img, cx - text_w(text, size) // 2, y, text, size, color)


def in_rect(x, y, r):
    return r[0] <= x < r[0] + r[2] and r[1] <= y < r[1] + r[3]


def write_dets_json(path, res, shown, sel=ALL_IDX):
    """把这一帧的框写成 json（只记当前筛选类别），供历史详情页回看"""
    items = []
    try:
        if res:
            bx = res[0]
            cl = res[1]
            cf = res[2]
            for i in range(len(bx)):
                c = int(cl[i])
                if sel != ALL_IDX and c != sel:
                    continue
                q = bx[i]
                items.append('{"cls": %d, "conf": %.3f, "x": %d, "y": %d, "w": %d, "h": %d}'
                             % (c, float(cf[i]), int(q[0]), int(q[1]),
                                int(q[2]), int(q[3])))
        f = open(path, "w")
        f.write('{"count": %d, "detections": [' % len(items))
        f.write(", ".join(items))
        f.write("]}\n")
        f.close()
        return len(items)
    except Exception as e:
        print("[UI] json fail", e)
        return -1


def load_boxes(path):
    """读回 [{cls,conf,x,y,w,h}, ...]，失败返回 []"""
    out = []
    try:
        import json
        f = open(path)
        txt = f.read()
        f.close()
        d = json.loads(txt)
        for q in d["detections"]:
            out.append((int(q["cls"]), float(q["conf"]), int(q["x"]),
                        int(q["y"]), int(q["w"]), int(q["h"])))
    except Exception:
        return []
    return out


def sd_free_mb():
    for p in ("/sdcard", "/data"):
        try:
            st = os.statvfs(p)
            return int(st[0] * st[3] / (1024 * 1024))
        except Exception:
            pass
    return -1


def pick_save_dir():
    for d in SAVE_DIRS:
        try:
            os.stat(d)
            return d
        except Exception:
            try:
                os.mkdir(d)
                return d
            except Exception:
                pass
    return "/data"


def next_seq(d):
    seq = 1
    try:
        names = os.listdir(d)
    except Exception:
        names = []
    for n in names:
        if n.startswith("IMG_") and n.endswith(".jpg"):
            try:
                v = int(n[4:8])
                if v >= seq:
                    seq = v + 1
            except Exception:
                pass
    return seq


def add_history(seq, name, cnt, fps, fname):
    try:
        f = open(HISTORY_PATH, "a")
        f.write("%d|%s|%d|%.1f|%s\n" % (seq, name, cnt, fps, fname))
        f.close()
    except Exception as e:
        print("[UI] history write fail", e)


def read_history(n=9):
    try:
        f = open(HISTORY_PATH)
        lines = f.readlines()
        f.close()
    except Exception:
        return []
    out = []
    for ln in lines[-n:]:
        out.append(ln.strip())
    return out


def clear_history():
    try:
        os.remove(HISTORY_PATH)
        return True
    except Exception:
        return False


# ---------------- 触摸 ----------------
class Touch:
    def __init__(self):
        self.ok = False
        self.last_ms = 0
        self.down = False
        self.buf = bytearray(1)
        self.val = bytearray(8)
        try:
            self.i2c = I2C(TOUCH_I2C_ID, freq=400000)
            self.ok = TOUCH_ADDR in self.i2c.scan()
        except Exception as e:
            print("[UI] touch init fail", e)

    def read_raw(self):
        try:
            self.buf[0] = 0x00
            self.i2c.writeto(TOUCH_ADDR, self.buf)
            self.i2c.readfrom_into(TOUCH_ADDR, self.val)
        except Exception:
            return None
        d = self.val
        cnt = d[2] & 0x0F
        if cnt < 1:
            return None
        rx = ((d[3] & 0x0F) << 8) | d[4]
        ry = ((d[5] & 0x0F) << 8) | d[6]
        ev = d[3] >> 6
        return (rx, ry, ev)

    def to_disp(self, rx, ry):
        x = TM_A * rx + TM_B * ry + TM_C
        y = TM_D * rx + TM_E * ry + TM_F
        if x < 0:
            x = 0
        elif x > DISPLAY_SIZE[0] - 1:
            x = DISPLAY_SIZE[0] - 1
        if y < 0:
            y = 0
        elif y > DISPLAY_SIZE[1] - 1:
            y = DISPLAY_SIZE[1] - 1
        return int(x), int(y)

    def tap(self):
        """按下沿触发：FT5x06 的 ev==1 是抬起，30 Hz 轮询基本采不到，
        所以只要检测到接触（ev 0 或 2）且上一轮没有接触，就算一次点击。"""
        if not self.ok:
            return None
        r = self.read_raw()
        if r is None:
            self.down = False
            return None
        rx, ry, ev = r
        if ev == 1:
            self.down = False
            return None
        if self.down:
            return None
        now = time.ticks_ms()
        if time.ticks_diff(now, self.last_ms) < TOUCH_REARM_MS:
            return None
        self.down = True
        self.last_ms = now
        return self.to_disp(rx, ry)


# ---------------- 画框 ----------------
def draw_boxes(img, res, sel, thickness=2, text_size=20, draw_label=True):
    if res is None:
        return 0
    n = 0
    boxes = res[0]
    clss = res[1]
    confs = res[2]
    total = len(boxes)
    dense = total > 18
    for i in range(total):
        ci = int(clss[i])
        if ci < 0 or ci >= NC:
            continue
        if sel != ALL_IDX and ci != sel:
            continue
        b = boxes[i]
        x = int(b[0])
        y = int(b[1])
        w = int(b[2])
        h = int(b[3])
        c = COLORS[ci]
        img.draw_rectangle(x, y, w, h, color=c, thickness=thickness)
        if draw_label and (not dense or confs[i] > 0.55):
            ty = y - text_size - 2
            if ty < 2:
                ty = y + 2
            text_at(img, x, ty, "%s %.2f" % (LABELS[ci], confs[i]), text_size, c)
        n += 1
    return n


# ---------------- 页面绘制 ----------------
def draw_home(osd, free_kb, sd_mb, n_hist, armed=False):
    osd.clear()
    osd.draw_rectangle(0, 0, DISPLAY_SIZE[0], DISPLAY_SIZE[1], color=BG, fill=True)
    osd.draw_rectangle(0, 0, DISPLAY_SIZE[0], 4, color=CYAN, fill=True)
    osd.draw_rectangle(0, 4, DISPLAY_SIZE[0], 2, color=(0, 90, 110), fill=True)

    text_center(osd, 400, 16, "K230 边缘视觉", 40, WHITE)
    osd.draw_rectangle(206, 68, 388, 26, color=CARD, fill=True)
    text_center(osd, 400, 73, "20 类地物   416x416   触摸开始", 15, GREY)

    osd.draw_rectangle(EXIT_RECT[0], EXIT_RECT[1], EXIT_RECT[2], EXIT_RECT[3], color=(90, 34, 34) if armed else CARD, fill=True)
    osd.draw_rectangle(EXIT_RECT[0], EXIT_RECT[1], EXIT_RECT[2], EXIT_RECT[3], color=RED if armed else LINE,
                       thickness=2 if armed else 1)
    text_center(osd, EXIT_RECT[0] + EXIT_RECT[2] // 2, EXIT_RECT[1] + (EXIT_RECT[3] - 20) // 2, "退出", 20, WHITE)
    osd.draw_rectangle(40, 106, 720, 1, color=LINE, fill=True)

    for i in range(len(HOME_BTNS)):
        name, sub, r = HOME_BTNS[i]
        acc = ACCENTS[i]
        osd.draw_rectangle(r[0], r[1], r[2], r[3], color=CARD, fill=True)
        osd.draw_rectangle(r[0], r[1], r[2], r[3], color=LINE, thickness=1)
        osd.draw_rectangle(r[0], r[1], 6, r[3], color=acc, fill=True)
        osd.draw_rectangle(r[0] + 18, r[1] + 20, 54, 54, color=(28, 30, 42), fill=True)
        osd.draw_rectangle(r[0] + 18, r[1] + 20, 54, 54, color=acc, thickness=2)
        text_center(osd, r[0] + 45, r[1] + 30, "0%d" % (i + 1), 26, acc)
        text_at(osd, r[0] + 88, r[1] + 16, name, 26, WHITE)
        text_at(osd, r[0] + 88, r[1] + 56, sub, 15, GREY)
        text_at(osd, r[0] + r[2] - 30, r[1] + r[3] // 2 - 14, ">", 24, acc)

    osd.draw_rectangle(40, 418, 720, 1, color=LINE, fill=True)
    text_at(osd, 42, 430, "空闲", 14, GREY)
    text_at(osd, 84, 424, "%d KB" % free_kb, 20, CYAN)
    text_at(osd, 244, 430, "存储卡", 14, GREY)
    text_at(osd, 298, 424, "%d MB" % sd_mb, 20, CYAN)
    text_at(osd, 424, 430, "历史", 14, GREY)
    text_at(osd, 466, 424, "%d 条" % n_hist, 20, CYAN)
    text_center(osd, 640, 430, "点右上角 退出 两次", 14, GREY)


def draw_class_page(osd, sel):
    osd.clear()
    osd.draw_rectangle(0, 0, DISPLAY_SIZE[0], DISPLAY_SIZE[1], color=BG, fill=True)
    text_at(osd, 16, 12, "选择识别类别", 26, WHITE)
    osd.draw_rectangle(BACK_RECT[0], BACK_RECT[1], BACK_RECT[2], BACK_RECT[3], color=CARD_HI, fill=True)
    text_center(osd, BACK_RECT[0] + BACK_RECT[2] // 2, BACK_RECT[1] + (BACK_RECT[3] - 22) // 2, "返回", 22, WHITE)

    cw = 148
    ch = 74
    x0 = 14
    y0 = 58
    gap_x = 6
    gap_y = 6
    for i in range(NC):
        r = i // 5
        c = i % 5
        x = x0 + c * (cw + gap_x)
        y = y0 + r * (ch + gap_y)
        on = (sel == i)
        osd.draw_rectangle(x, y, cw, ch, color=CARD_HI if on else CARD, fill=True)
        osd.draw_rectangle(x, y, cw, ch, color=COLORS[i] if on else LINE,
                           thickness=2 if on else 1)
        osd.draw_rectangle(x + 10, y + 10, 16, 16, color=COLORS[i], fill=True)
        text_at(osd, x + 34, y + 6, CNNAME[i], 20, WHITE)
        text_at(osd, x + 34, y + 34, LABELS[i], 13, GREY)
        text_at(osd, x + 34, y + 54, "#%02d" % i, 12, GREY)

    ay = y0 + 4 * (ch + gap_y) + 4
    osd.draw_rectangle(14, ay, 772, 56, color=CARD_HI, fill=True)
    osd.draw_rectangle(14, ay, 772, 56, color=YELLOW, thickness=2)
    text_center(osd, 400, ay + 14, "全部类别一起识别", 24, YELLOW)


def draw_guide(osd, gp=0):
    osd.clear()
    osd.draw_rectangle(0, 0, DISPLAY_SIZE[0], DISPLAY_SIZE[1], color=BG, fill=True)
    osd.draw_rectangle(0, 0, DISPLAY_SIZE[0], 4, color=CYAN, fill=True)
    text_at(osd, 16, 12, "识别结果说明", 26, WHITE)
    text_at(osd, 400, 18, "%d / 2" % (gp + 1), 18, CYAN)
    osd.draw_rectangle(BACK_RECT[0], BACK_RECT[1], BACK_RECT[2], BACK_RECT[3], color=CARD_HI, fill=True)
    text_center(osd, BACK_RECT[0] + BACK_RECT[2] // 2, BACK_RECT[1] + (BACK_RECT[3] - 22) // 2, "返回", 22, WHITE)
    osd.draw_rectangle(12, 56, 776, 1, color=LINE, fill=True)

    first = gp * 10
    for k in range(10):
        i = first + k
        if i >= NC:
            break
        col = k // 5
        row = k % 5
        x = 18 + col * 392
        y = 66 + row * 68
        osd.draw_rectangle(x, y + 8, 20, 20, color=COLORS[i], fill=True)
        text_at(osd, x + 30, y, CNNAME[i], 23, WHITE)
        text_at(osd, x + 30, y + 30, LABELS[i], 15, CYAN)
        text_at(osd, x + 30, y + 50, NOTE[i], 14, GREY)

    osd.draw_rectangle(20, 420, 160, 46, color=CARD_HI if gp > 0 else CARD, fill=True)
    text_center(osd, 100, 434, "上一页", 20, WHITE if gp > 0 else GREY)
    osd.draw_rectangle(620, 420, 160, 46, color=CARD_HI if gp < 1 else CARD, fill=True)
    text_center(osd, 700, 434, "下一页", 20, WHITE if gp < 1 else GREY)


def draw_usage(osd, sd_mb):
    osd.clear()
    osd.draw_rectangle(0, 0, DISPLAY_SIZE[0], DISPLAY_SIZE[1], color=BG, fill=True)
    osd.draw_rectangle(0, 0, DISPLAY_SIZE[0], 4, color=CYAN, fill=True)
    text_at(osd, 16, 10, "设备使用说明", 26, WHITE)
    osd.draw_rectangle(BACK_RECT[0], BACK_RECT[1], BACK_RECT[2], BACK_RECT[3], color=CARD_HI, fill=True)
    text_center(osd, BACK_RECT[0] + BACK_RECT[2] // 2, BACK_RECT[1] + (BACK_RECT[3] - 22) // 2, "返回", 22, WHITE)
    osd.draw_rectangle(12, 52, 776, 1, color=LINE, fill=True)
    lines = [
        "1. 主页点 识别内容选择 进入类别列表",
        "2. 点一个类别，进入实时识别画面",
        "3. 底部 拍照 存一张带框照片和坐标",
        "4.  菜单 回类别列表，再 返回 回主页",
        "5. 右上角 退出 连点两次结束程序",
        "6. 照片存到 /sdcard/photo/IMG_xxxx.jpg",
        "7. 主页 识别历史记录 可回看每次结果",
        "8.  系统与清理 可清内存、看剩余空间",
        "",
        "拍摄要点：",
        "- 镜头对准航拍图：屏幕上、打印件或纸质地图",
        "- 对着桌面拍没有目标，画面会是空的",
        "- 视野越紧目标越大，检出率越高；12 像素是极限",
        "- 避免屏幕反光和摩尔纹",
    ]
    for i in range(len(lines)):
        text_at(osd, 24, 52 + i * 26, lines[i], 17, WHITE if lines[i] else GREY)
    text_at(osd, 24, 428, "存储卡剩余：%d MB" % sd_mb, 17, GREY)


def draw_history(osd, rows, hp=0):
    osd.clear()
    osd.draw_rectangle(0, 0, DISPLAY_SIZE[0], DISPLAY_SIZE[1], color=BG, fill=True)
    osd.draw_rectangle(0, 0, DISPLAY_SIZE[0], 4, color=CYAN, fill=True)
    text_at(osd, 16, 12, "识别历史记录", 26, WHITE)
    osd.draw_rectangle(BACK_RECT[0], BACK_RECT[1], BACK_RECT[2], BACK_RECT[3], color=CARD_HI, fill=True)
    text_center(osd, BACK_RECT[0] + BACK_RECT[2] // 2, BACK_RECT[1] + (BACK_RECT[3] - 22) // 2, "返回", 22, WHITE)
    osd.draw_rectangle(12, 56, 776, 1, color=LINE, fill=True)

    tot = len(rows)
    objs = 0
    classes = {}
    for line in rows:
        q = line.split("|")
        if len(q) < 5:
            continue
        try:
            objs += int(q[2])
        except Exception:
            pass
        classes[q[1]] = 1
    text_at(osd, 18, 64, "%d 条记录" % tot, 17, CYAN)
    text_at(osd, 200, 64, "%d 个类别" % len(classes), 17, CYAN)
    text_at(osd, 382, 64, "共 %d 个目标" % objs, 17, CYAN)
    text_at(osd, 610, 64, "点一行看坐标", 15, GREY)

    if not rows:
        text_center(osd, 400, 210, "还没有记录", 24, GREY)
        text_center(osd, 400, 252, "识别后点 拍照 就会记一条", 17, GREY)
        return

    npages = (len(rows) + 2) // 3
    if hp >= npages:
        hp = npages - 1
    for i in range(3):
        k = hp * 3 + i
        if k >= len(rows):
            break
        q = rows[len(rows) - 1 - k].split("|")
        if len(q) < 5:
            continue
        y = 96 + i * 100
        osd.draw_rectangle(14, y, 772, 92, color=CARD, fill=True)
        osd.draw_rectangle(14, y, 772, 92, color=LINE, thickness=1)
        osd.draw_rectangle(14, y, 6, 92, color=CYAN, fill=True)
        osd.draw_rectangle(30, y + 16, 60, 60, color=(28, 30, 42), fill=True)
        osd.draw_rectangle(30, y + 16, 60, 60, color=CYAN, thickness=2)
        text_center(osd, 60, y + 30, "%02d" % (int(q[0]) % 100), 26, CYAN)
        text_at(osd, 106, y + 18, cn_of(q[1]), 26, WHITE)
        text_at(osd, 106, y + 56, "%s 个目标   %s fps" % (q[2], q[3]), 18, GREY)
        text_at(osd, 430, y + 20, q[4], 20, CYAN)
        text_at(osd, 430, y + 52, "点开看坐标", 15, GREY)
        text_at(osd, 744, y + 30, ">", 26, CYAN)

    osd.draw_rectangle(20, 420, 150, 46, color=CARD_HI if hp > 0 else CARD, fill=True)
    text_center(osd, 95, 434, "上一页", 19, WHITE if hp > 0 else GREY)
    text_center(osd, 400, 434, "第 %d / %d 页" % (hp + 1, npages), 18, GREY)
    osd.draw_rectangle(630, 420, 150, 46, color=CARD_HI if hp < npages - 1 else CARD, fill=True)
    text_center(osd, 705, 434, "下一页", 19, WHITE if hp < npages - 1 else GREY)


def draw_detail(osd, rec, boxes, idx, total):
    osd.clear()
    osd.draw_rectangle(0, 0, DISPLAY_SIZE[0], DISPLAY_SIZE[1], color=BG, fill=True)
    osd.draw_rectangle(0, 0, DISPLAY_SIZE[0], 4, color=CYAN, fill=True)
    q = rec.split("|")
    if len(q) < 5:
        q = ["0", "-", "0", "0", "-"]
    text_at(osd, 16, 12, "记录 %s" % q[0], 26, WHITE)
    osd.draw_rectangle(BACK_RECT[0], BACK_RECT[1], BACK_RECT[2], BACK_RECT[3], color=CARD_HI, fill=True)
    text_center(osd, BACK_RECT[0] + BACK_RECT[2] // 2, BACK_RECT[1] + (BACK_RECT[3] - 22) // 2, "返回", 22, WHITE)
    text_at(osd, 16, 60, "%s   %s 个目标   %s fps   %s" % (cn_of(q[1]), q[2], q[3], q[4]),
            17, CYAN)
    osd.draw_rectangle(12, 86, 776, 1, color=LINE, fill=True)

    # 左侧：迷你地图（800x480 等比缩到 372x223）
    mx = 18
    my = 100
    mw = 372
    mh = 223
    osd.draw_rectangle(mx, my, mw, mh, color=(10, 12, 18), fill=True)
    osd.draw_rectangle(mx, my, mw, mh, color=LINE, thickness=1)
    sx = mw / 800.0
    sy = mh / 480.0
    text_at(osd, mx + 6, my + 4, "框位置", 13, GREY)
    if not boxes:
        text_center(osd, mx + mw // 2, my + mh // 2 - 8, "没有坐标数据", 16, GREY)
    else:
        for b in boxes:
            bx = int(b[2] * sx) + mx
            by = int(b[3] * sy) + my
            bw = int(b[4] * sx)
            bh = int(b[5] * sy)
            if bw < 2:
                bw = 2
            if bh < 2:
                bh = 2
            osd.draw_rectangle(bx, by, bw, bh, color=COLORS[b[0] % len(COLORS)], thickness=1)

    # 右侧：坐标列表
    text_at(osd, 410, 100, "坐标列表", 16, GREY)
    text_at(osd, 410, 124, "x    y    w    h   置信度", 14, GREY)
    n = len(boxes)
    if n > 9:
        n = 9
    for i in range(n):
        b = boxes[i]
        y = 148 + i * 26
        if i % 2 == 0:
            osd.draw_rectangle(404, y - 3, 380, 24, color=CARD, fill=True)
        text_at(osd, 412, y, "%3d  %3d  %2d  %2d   %.2f"
                % (b[2], b[3], b[4], b[5], b[1]), 16, WHITE)
    if len(boxes) > 9:
        text_at(osd, 412, 148 + 9 * 26, "还有 %d 个" % (len(boxes) - 9), 14, GREY)

    osd.draw_rectangle(20, 420, 150, 46, color=CARD_HI if idx < total - 1 else CARD, fill=True)
    text_center(osd, 95, 434, "上一个", 19, WHITE if idx < total - 1 else GREY)
    text_center(osd, 400, 434, "%d / %d" % (idx + 1, total), 18, GREY)
    osd.draw_rectangle(630, 420, 150, 46, color=CARD_HI if idx > 0 else CARD, fill=True)
    text_center(osd, 705, 434, "下一个", 19, WHITE if idx > 0 else GREY)


def draw_system(osd, info):
    osd.clear()
    osd.draw_rectangle(0, 0, DISPLAY_SIZE[0], DISPLAY_SIZE[1], color=BG, fill=True)
    osd.draw_rectangle(0, 0, DISPLAY_SIZE[0], 4, color=CYAN, fill=True)
    text_at(osd, 16, 10, "系统与清理", 26, WHITE)
    osd.draw_rectangle(BACK_RECT[0], BACK_RECT[1], BACK_RECT[2], BACK_RECT[3], color=CARD_HI, fill=True)
    text_center(osd, BACK_RECT[0] + BACK_RECT[2] // 2, BACK_RECT[1] + (BACK_RECT[3] - 22) // 2, "返回", 22, WHITE)
    osd.draw_rectangle(12, 52, 776, 1, color=LINE, fill=True)
    for i in range(len(info)):
        text_at(osd, 24, 72 + i * 36, info[i], 20, WHITE if i < 2 else GREY)
    osd.draw_rectangle(20, 420, 260, 46, color=(30, 80, 50), fill=True)
    osd.draw_rectangle(20, 420, 260, 46, color=GREEN, thickness=2)
    text_center(osd, 150, 434, "清理内存", 22, WHITE)


def draw_detect_bar(osd, sel, shown, fps_v, toast):
    osd.draw_rectangle(0, 0, DISPLAY_SIZE[0], 34, color=(0, 0, 0), fill=True)
    nm = "全部" if sel == ALL_IDX else CNNAME[sel]
    c = YELLOW if sel == ALL_IDX else COLORS[sel]
    text_at(osd, 8, 4, nm, 24, c)
    text_at(osd, 250, 6, "%d 个目标" % shown, 20, WHITE)
    text_at(osd, DISPLAY_SIZE[0] - 130, 6, "%.1f fps" % fps_v, 20, GREEN)
    by = 414
    osd.draw_rectangle(PHOTO_RECT[0], PHOTO_RECT[1], PHOTO_RECT[2], PHOTO_RECT[3], color=(26, 70, 40), fill=True)
    osd.draw_rectangle(PHOTO_RECT[0], PHOTO_RECT[1], PHOTO_RECT[2], PHOTO_RECT[3], color=GREEN, thickness=2)
    text_center(osd, PHOTO_RECT[0] + PHOTO_RECT[2] // 2, PHOTO_RECT[1] + (PHOTO_RECT[3] - 22) // 2, "拍照", 22, WHITE)
    osd.draw_rectangle(MENU_RECT[0], MENU_RECT[1], MENU_RECT[2], MENU_RECT[3], color=(70, 30, 30), fill=True)
    osd.draw_rectangle(MENU_RECT[0], MENU_RECT[1], MENU_RECT[2], MENU_RECT[3], color=RED, thickness=2)
    text_center(osd, MENU_RECT[0] + MENU_RECT[2] // 2, MENU_RECT[1] + (MENU_RECT[3] - 22) // 2, "菜单", 22, WHITE)
    if toast:
        w = text_w(toast, 20) + 40
        osd.draw_rectangle(400 - w // 2, 360, w, 40, color=(0, 0, 0), fill=True)
        osd.draw_rectangle(400 - w // 2, 360, w, 40, color=CYAN, thickness=1)
        text_center(osd, 400, 368, toast, 20, CYAN)


def system_info(clean_note):
    out = [
        "空闲内存：%d KB" % (gc.mem_free() // 1024),
        "存储卡剩余：%d MB" % sd_free_mb(),
        "模型：20 类地物，416x416 输入",
        "模型大小：%d KB" % (3573584 // 1024),
        "触摸：i2c%d 地址 0x%02X" % (TOUCH_I2C_ID, TOUCH_ADDR),
        "上次清理：%s" % (clean_note if clean_note else "-"),
        "历史文件：/sdcard/det_history.txt",
    ]
    return out


def main():
    fpioa = FPIOA()
    fpioa.set_function(BUTTON_PIN, FPIOA.GPIO64)
    fpioa.set_function(LED_R_PIN, FPIOA.GPIO65)
    fpioa.set_function(LED_G_PIN, FPIOA.GPIO66)
    key = Pin(BUTTON_PIN, Pin.IN, Pin.PULL_DOWN)
    led_r = Pin(LED_R_PIN, Pin.OUT, pull=Pin.PULL_NONE, drive=7)
    led_g = Pin(LED_G_PIN, Pin.OUT, pull=Pin.PULL_NONE, drive=7)
    led_r.value(1)
    led_g.value(0)

    sensor = None
    yolo = None
    osd = None
    touch = Touch()
    page = P_HOME
    sel = ALL_IDX
    seq = 1
    save_dir = "/sdcard/photo"
    msg_text = ""
    msg_until = 0
    fps_v = 0.0
    clean_note = ""
    hist_rows = []
    quit_armed = False
    guide_page = 0
    hist_page = 0
    det_idx = 0
    hist_tick = [0]
    key_state = [0]

    def key_edge():
        """只在 0->1 上升沿返回 True；引脚若被拉死为高则永不触发"""
        cur = 1 if key.value() == KEY_PRESSED else 0
        prev = key_state[0]
        key_state[0] = cur
        return prev == 0 and cur == 1

    def hold_key():
        """轮询 USR，返回按住的毫秒数"""
        time.sleep_ms(DEBOUNCE_MS)
        t0 = time.ticks_ms()
        held = 0
        while key.value() == KEY_PRESSED:
            held = time.ticks_diff(time.ticks_ms(), t0)
            if held >= HOLD_MS:
                break
            time.sleep_ms(20)
        return held

    def take_photo(res, shown):
        """存一张带框照片 + 记历史，返回新 seq"""
        photo = sensor.snapshot(chn=CAM_CHN_ID_1)
        draw_boxes(photo, res, sel, 2, 22, False)
        fname = "%s/IMG_%04d.jpg" % (save_dir, seq)
        okf = False
        try:
            photo.save(fname, quality=JPEG_QUALITY)
            okf = True
        except Exception as e:
            print("[UI] save fail", e)
        photo = None
        gc.collect()
        if okf:
            write_dets_json(fname[:-4] + ".json", res, shown, sel)
            add_history(seq, "ALL" if sel == ALL_IDX else LABELS[sel], shown, fps_v,
                        "IMG_%04d.jpg" % seq)
            print("[UI] saved %s (%d objs)" % (fname, shown))
            return seq + 1, "已保存 IMG_%04d.jpg" % seq
        return seq, "保存失败"

    try:
        print("[UI] boot")
        dbg("boot")
        for fn in (MediaManager.deinit, Sensor.deinit, Display.deinit):
            try:
                fn()
            except Exception:
                pass

        print("[UI] key=%d after gpio" % key.value())
        sensor = Sensor(fps=30)
        sensor.reset()
        sensor.set_framesize(w=DISPLAY_SIZE[0], h=DISPLAY_SIZE[1], chn=CAM_CHN_ID_0)
        sensor.set_pixformat(Sensor.YUV420SP, chn=CAM_CHN_ID_0)
        sensor.set_framesize(w=DISPLAY_SIZE[0], h=DISPLAY_SIZE[1], chn=CAM_CHN_ID_1)
        sensor.set_pixformat(Sensor.RGB565, chn=CAM_CHN_ID_1)
        sensor.set_framesize(w=RGB888P_SIZE[0], h=RGB888P_SIZE[1], chn=CAM_CHN_ID_2)
        sensor.set_pixformat(Sensor.RGBP888, chn=CAM_CHN_ID_2)

        print("[UI] key=%d after sensor cfg" % key.value())
        Display.init(Display.ST7701, width=DISPLAY_SIZE[0], height=DISPLAY_SIZE[1],
                     osd_num=1, to_ide=True)
        print("[UI] key=%d after display init" % key.value())
        osd = image.Image(DISPLAY_SIZE[0], DISPLAY_SIZE[1], image.ARGB8888)
        Display.bind_layer(**sensor.bind_info(x=0, y=0, chn=CAM_CHN_ID_0),
                           layer=Display.LAYER_VIDEO1)
        print("[UI] key=%d after bind_layer" % key.value())
        MediaManager.init()
        print("[UI] key=%d after media init" % key.value())
        sensor.run()
        print("[UI] key=%d after sensor run" % key.value())
        dbg("sensor run")

        # ---- 开机动画 ----
        osd.clear()
        osd.draw_rectangle(0, 0, DISPLAY_SIZE[0], DISPLAY_SIZE[1], color=(8, 9, 14), fill=True)
        Display.show_image(osd, 0, 0, Display.LAYER_OSD3)
        time.sleep_ms(120)
        dbg("blank")
        text_center(osd, 400, 188, "K230", 60, WHITE)
        Display.show_image(osd, 0, 0, Display.LAYER_OSD3)
        time.sleep_ms(240)
        dbg("title1")
        text_center(osd, 400, 266, "边缘视觉", 28, CYAN)
        Display.show_image(osd, 0, 0, Display.LAYER_OSD3)
        time.sleep_ms(240)
        dbg("title2")
        for k in range(0, 21):
            osd.draw_rectangle(250, 328, 300, 6, color=(30, 33, 44), fill=True)
            osd.draw_rectangle(250, 328, 15 * k, 6, color=CYAN, fill=True)
            Display.show_image(osd, 0, 0, Display.LAYER_OSD3)
            time.sleep_ms(40)
        dbg("bar done")
        text_center(osd, 400, 350, "正在加载 20 类模型", 16, GREY)
        Display.show_image(osd, 0, 0, Display.LAYER_OSD3)

        # ---- 载入模型 ----
        dbg("model new")
        yolo = YOLOv8(task_type="detect", mode="video", kmodel_path=KMODEL_PATH,
                      labels=LABELS, rgb888p_size=RGB888P_SIZE,
                      model_input_size=MODEL_INPUT_SIZE,
                      display_size=DISPLAY_SIZE, conf_thresh=CONF_THRESH,
                      nms_thresh=NMS_THRESH, max_boxes_num=MAX_BOXES, debug_mode=0)
        dbg("model built")
        yolo.config_preprocess()
        dbg("preprocess ok")
        gc.collect()
        print("[UI] key=%d after model load" % key.value())
        fpioa.set_function(BUTTON_PIN, FPIOA.GPIO64)
        key = Pin(BUTTON_PIN, Pin.IN, Pin.PULL_DOWN)
        print("[UI] key=%d after pin reinit" % key.value())
        dbg("pin reinit")
        save_dir = pick_save_dir()
        seq = next_seq(save_dir)
        dbg("seq %d" % seq)
        print("[UI] ready, free=%dKB sd=%dMB touch=%s seq=%d" % (
            gc.mem_free() // 1024, sd_free_mb(), touch.ok, seq))

        # 等用户松手（上一次退出时可能还按着 USR，避免一进来就判长按）
        t_wait = time.ticks_ms()
        while key.value() == KEY_PRESSED and time.ticks_diff(time.ticks_ms(), t_wait) < 6000:
            time.sleep_ms(30)
        time.sleep_ms(150)

        page = P_HOME
        fps_t0 = time.ticks_ms()
        fps_n = 0
        frame_i = 0
        res_keep = None
        shown_keep = 0

        last_page = -1
        while True:
            if page != last_page:
                print("[UI] page -> %d" % page)
                last_page = page
            tap = touch.tap()

            # ---------- 主页 ----------
            if page == P_HOME:
                if tap:
                    print("[UI] tap home %d,%d" % (tap[0], tap[1]))
                    if in_rect(tap[0], tap[1], EXIT_RECT):
                        if quit_armed:
                            print("[UI] quit confirmed")
                            break
                        quit_armed = True
                        continue
                    quit_armed = False
                    for i in range(len(HOME_BTNS)):
                        if in_rect(tap[0], tap[1], HOME_BTNS[i][2]):
                            if i == 0:
                                page = P_CLASS
                            elif i == 1:
                                hist_rows = read_history(9)
                                page = P_HISTORY
                            elif i == 2:
                                page = P_GUIDE
                            elif i == 3:
                                page = P_USAGE
                            else:
                                page = P_SYSTEM
                            break
                    continue
                draw_home(osd, gc.mem_free() // 1024, sd_free_mb(),
                          len(read_history(1)), quit_armed)
                Display.show_image(osd, 0, 0, Display.LAYER_OSD3)
                time.sleep_ms(25)
                continue

            # ---------- 类别选择 ----------
            if page == P_CLASS:
                if tap:
                    if in_rect(tap[0], tap[1], BACK_RECT):
                        page = P_HOME
                        continue
                    cw, ch, x0, y0, gx, gy = 148, 74, 14, 58, 6, 6
                    hit = False
                    for i in range(NC):
                        r = i // 5
                        c = i % 5
                        if in_rect(tap[0], tap[1],
                                   (x0 + c * (cw + gx), y0 + r * (ch + gy), cw, ch)):
                            sel = i
                            page = P_DETECT
                            fps_t0 = time.ticks_ms()
                            fps_n = 0
                            print("[UI] select %s" % LABELS[i])
                            hit = True
                            break
                    if not hit:
                        ay = y0 + 4 * (ch + gy) + 4
                        if in_rect(tap[0], tap[1], (14, ay, 772, 56)):
                            sel = ALL_IDX
                            page = P_DETECT
                            fps_t0 = time.ticks_ms()
                            fps_n = 0
                            print("[UI] select ALL")
                    continue
                draw_class_page(osd, sel)
                Display.show_image(osd, 0, 0, Display.LAYER_OSD3)
                time.sleep_ms(25)
                continue

            # ---------- 说明 / 用法 / 系统 ----------
            if page in (P_GUIDE, P_USAGE, P_SYSTEM):
                if tap and in_rect(tap[0], tap[1], BACK_RECT):
                    page = P_HOME
                    continue
                if page == P_SYSTEM and tap and in_rect(tap[0], tap[1], (20, 420, 260, 46)):
                    f0 = gc.mem_free() // 1024
                    gc.collect()
                    gc.collect()
                    f1 = gc.mem_free() // 1024
                    clean_note = "%d -> %d KB" % (f0, f1)
                    print("[UI] clean memory", clean_note)
                    continue
                if page == P_GUIDE and tap:
                    if in_rect(tap[0], tap[1], (20, 420, 160, 46)):
                        if guide_page > 0:
                            guide_page -= 1
                        continue
                    if in_rect(tap[0], tap[1], (620, 420, 160, 46)):
                        if guide_page < 1:
                            guide_page += 1
                        continue
                if page == P_GUIDE:
                    draw_guide(osd, guide_page)
                elif page == P_USAGE:
                    draw_usage(osd, sd_free_mb())
                else:
                    draw_system(osd, system_info(clean_note))
                Display.show_image(osd, 0, 0, Display.LAYER_OSD3)
                time.sleep_ms(25)
                continue

            # ---------- 历史 ----------
            if page == P_HISTORY:
                if tap:
                    if in_rect(tap[0], tap[1], BACK_RECT):
                        page = P_HOME
                        continue
                    if in_rect(tap[0], tap[1], (20, 420, 150, 46)):
                        if hist_page > 0:
                            hist_page -= 1
                        continue
                    if in_rect(tap[0], tap[1], (630, 420, 150, 46)):
                        npages = (len(hist_rows) + 2) // 3
                        if hist_page < npages - 1:
                            hist_page += 1
                        continue
                    for i in range(3):
                        ry = 96 + i * 100
                        if in_rect(tap[0], tap[1], (14, ry, 772, 92)):
                            k = hist_page * 3 + i
                            if k < len(hist_rows):
                                det_idx = len(hist_rows) - 1 - k
                                page = P_DETAIL
                                print("[UI] open record %d" % det_idx)
                            break
                    continue
                hist_tick[0] += 1
                if hist_tick[0] % 20 == 1 or not hist_rows:
                    hist_rows = read_history(60)
                draw_history(osd, hist_rows, hist_page)
                Display.show_image(osd, 0, 0, Display.LAYER_OSD3)
                time.sleep_ms(25)
                continue

            # ---------- 记录详情 ----------
            if page == P_DETAIL:
                if tap:
                    if in_rect(tap[0], tap[1], BACK_RECT):
                        page = P_HISTORY
                        continue
                    if in_rect(tap[0], tap[1], (20, 420, 150, 46)):
                        if det_idx < len(hist_rows) - 1:
                            det_idx += 1
                        continue
                    if in_rect(tap[0], tap[1], (630, 420, 150, 46)):
                        if det_idx > 0:
                            det_idx -= 1
                        continue
                if not hist_rows:
                    page = P_HISTORY
                    continue
                if det_idx >= len(hist_rows):
                    det_idx = len(hist_rows) - 1
                rec = hist_rows[det_idx]
                fname = rec.split("|")[4]
                boxes = load_boxes(save_dir + "/" + fname[:-4] + ".json")
                draw_detail(osd, rec, boxes, det_idx, len(hist_rows))
                Display.show_image(osd, 0, 0, Display.LAYER_OSD3)
                time.sleep_ms(25)
                continue

            # ---------- 检测 ----------
            if tap:
                if in_rect(tap[0], tap[1], MENU_RECT):
                    page = P_CLASS
                    continue
                if in_rect(tap[0], tap[1], PHOTO_RECT):
                    seq, msg_text = take_photo(res_keep, shown_keep)
                    msg_until = time.ticks_add(time.ticks_ms(), 2200)
                    continue
            if key_edge():
                print("[UI] usr short -> photo")
                seq, msg_text = take_photo(res_keep, shown_keep)
                msg_until = time.ticks_add(time.ticks_ms(), 2200)
                continue

            ai_img = sensor.snapshot(chn=CAM_CHN_ID_2)
            arr = ai_img.to_numpy_ref()
            res = None
            try:
                tens = yolo.preprocess(arr)
                outs = yolo.inference(tens)
                tens = None
                arr = None
                ai_img = None
                gc.collect()
                res = yolo.postprocess(outs)
                outs = None
            except MemoryError:
                print("[UI] malloc fail -> skip frame")
                res = None
                gc.collect()

            res_keep = res
            osd.clear()
            shown_keep = draw_boxes(osd, res, sel, 2, 20, True)
            fps_n += 1
            dt = time.ticks_diff(time.ticks_ms(), fps_t0)
            if dt >= 1000:
                fps_v = fps_n * 1000.0 / dt
                fps_n = 0
                fps_t0 = time.ticks_ms()
            if time.ticks_diff(msg_until, time.ticks_ms()) > 0:
                draw_detect_bar(osd, sel, shown_keep, fps_v, msg_text)
            else:
                draw_detect_bar(osd, sel, shown_keep, fps_v, "")
            Display.show_image(osd, 0, 0, Display.LAYER_OSD3)
            frame_i += 1
            if frame_i % GC_INTERVAL == 0:
                gc.collect()

    except KeyboardInterrupt:
        print("[UI] interrupted")
    except BaseException as e:
        print("[UI] exception:", e)
        try:
            osd.clear()
            osd.draw_rectangle(0, 0, DISPLAY_SIZE[0], DISPLAY_SIZE[1], color=(60, 0, 0), fill=True)
            text_at(osd, 20, 200, "ERROR", 40, WHITE)
            text_at(osd, 20, 260, str(e)[:60], 18, WHITE)
            Display.show_image(osd, 0, 0, Display.LAYER_OSD3)
            time.sleep_ms(1500)
        except Exception:
            pass
    finally:
        print("[UI] cleanup")
        try:
            led_r.value(1)
            led_g.value(0)
        except Exception:
            pass
        try:
            if yolo:
                yolo.deinit()
        except Exception:
            pass
        try:
            if sensor:
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
        gc.collect()
        print("[UI] clean exit")


main()
