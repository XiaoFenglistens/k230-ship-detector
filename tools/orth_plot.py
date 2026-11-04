# -*- coding: utf-8 -*-
"""正交实验三张图：8 组召回柱状图 + 目标像素/视角/亮度 三条趋势线。"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

RUN = r'D:\rs-yolo\runs\orth'
D = json.load(open(os.path.join(RUN, 'orth_summary.json'), encoding='utf-8'))
G = D['groups']

for fam in ('Microsoft YaHei', 'SimHei', 'DengXian'):
    try:
        f = font_manager.findfont(font_manager.FontProperties(family=fam), fallback_to_default=False)
        plt.rcParams['font.family'] = fam
        break
    except Exception:
        continue
plt.rcParams['axes.unicode_minus'] = False

sids = sorted(G, key=int)
lbl = ['%s' % s for s in sids]
rec = [G[s]['recall'] for s in sids]
pre = [G[s]['precision'] for s in sids]
sub = ['%s/%s/%s' % (G[s]['dist'], G[s]['view'], G[s]['light']) for s in sids]

fig = plt.figure(figsize=(16.5, 9.2))
gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1], hspace=0.42, wspace=0.26)

# 上：8 组柱状
ax = fig.add_subplot(gs[0, :])
x = range(len(sids))
b1 = ax.bar([i - 0.19 for i in x], rec, width=0.38, color='#2b6cb0', label='召回率 R')
b2 = ax.bar([i + 0.19 for i in x], pre, width=0.38, color='#f6ad55', label='精确率 P')
for i, r in enumerate(rec):
    ax.text(i - 0.19, r + 0.008, '%.3f' % r, ha='center', fontsize=9, color='#2b6cb0')
for i, p in enumerate(pre):
    ax.text(i + 0.19, p + 0.008, '%.3f' % p, ha='center', fontsize=9, color='#b7791f')
ax.set_xticks(list(x))
ax.set_xticklabels(['组 %s\n%s' % (s, sub[i]) for i, s in enumerate(sids)], fontsize=9)
ax.set_ylabel('指标')
ax.set_ylim(0, 0.72)
ax.grid(axis='y', ls=':', alpha=0.5)
ax.legend(loc='upper right', fontsize=10)
ax.set_title('拍屏正交实验：8 组（每组取按稳后的 2 对，GT 210/组；摄像头拍 2560×1440 屏上的船港图）', fontsize=13)

def trend(ax, key, xlabel, bins, title, fmt='%.1f'):
    xs, ys, ns = [], [], []
    for lo, hi, name in bins:
        pts = [G[s] for s in sids if lo <= G[s][key] < hi]
        if not pts:
            continue
        tp = sum(p['tp'] for p in pts)
        gt = sum(p['gt'] for p in pts)
        xs.append(sum(p[key] for p in pts) / len(pts))
        ys.append(tp / gt)
        ns.append(len(pts))
    ax.plot(xs, ys, 'o-', color='#2b6cb0', ms=9, lw=2)
    for xi, yi, ni in zip(xs, ys, ns):
        ax.annotate('%s\n(%d 组)' % (fmt % xi, ni), (xi, yi), textcoords='offset points',
                    xytext=(0, 11), ha='center', fontsize=8, color='#4a5568')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('召回率 R')
    ax.set_ylim(0, 0.45)
    ax.grid(ls=':', alpha=0.5)
    ax.set_title(title, fontsize=11)

trend(fig.add_subplot(gs[1, 0]), 'tgt_px', '画面里目标的中位边长（px）',
      [(0, 9, '<9'), (9, 11, '9-11'), (11, 13, '11-13'), (13, 99, '>13')],
      '① 目标占多少像素（距离）')
trend(fig.add_subplot(gs[1, 1]), 'trap_lr', '投影四边形左右边长比（1=正对）',
      [(0, 0.7, '<0.7'), (0.7, 0.85, '0.7-0.85'), (0.85, 99, '>0.85')],
      '② 拍摄角度（越偏越小）', fmt='%.2f')
trend(fig.add_subplot(gs[1, 2]), 'brightness', '画面平均亮度（0-255）',
      [(0, 60, '<60'), (60, 70, '60-70'), (70, 999, '>70')],
      '③ 画面亮度（光照）', fmt='%.0f')

out = os.path.join(RUN, 'orth_curve.png')
fig.savefig(out, dpi=110, bbox_inches='tight')
print('图 ->', out)
