# SPDX-FileCopyrightText: 2026 XiaoFenglistens
# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""从 yolo val 的日志里解析「逐类 AP 表」，画柱状图并写 JSON。

yolo val 的输出形如：
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                   all       5863      35439      0.712      0.585      0.641      0.442
               airplane        650        650      0.812      0.702      0.771      0.502
                airport        334        334      ...

用法：
    python tools\eval_multiclass_plot.py --log runs\dior20_416_val.log --out runs\dior20_416
"""
import argparse
import json
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROW = re.compile(
    r'^\s{2,}([A-Za-z][\w\-]*)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$')


def parse(log_path):
    rows = []
    ansi = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
    raw = open(log_path, 'rb').read()
    for enc in ('utf-8', 'utf-16', 'gbk'):
        try:
            text = raw.decode(enc)
            break
        except Exception:
            continue
    else:
        text = raw.decode('utf-8', errors='replace')
    for line in text.splitlines():
        line = ansi.sub('', line.replace('\r', ''))
        m = ROW.match(line)
        if not m:
            continue
        name, imgs, inst, p, r, m50, m5095 = m.groups()
        rows.append({
            'class': name, 'images': int(imgs), 'instances': int(inst),
            'precision': float(p), 'recall': float(r),
            'mAP50': float(m50), 'mAP50-95': float(m5095),
        })
    # 去重（日志可能被追加多次），保留最后一次出现的连续表
    seen = {}
    for r in rows:
        seen[r['class']] = r
    allrow = seen.pop('all', None)
    classes = sorted(seen.values(), key=lambda x: -x['mAP50'])
    return allrow, classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', required=True)
    ap.add_argument('--out', required=True, help='输出前缀，例如 runs\\dior20_416')
    ap.add_argument('--title', default='DIOR 20 类 · 逐类精度（YOLOv8n @416 · int8 之前的 fp32 验证）')
    args = ap.parse_args()

    allrow, classes = parse(args.log)
    if not classes:
        raise SystemExit('没解析到逐类结果，检查日志格式')
    print('总体:', allrow)
    print('%-28s %8s %8s' % ('class', 'mAP50', 'mAP50-95'))
    for c in classes:
        print('%-28s %8.3f %8.3f' % (c['class'], c['mAP50'], c['mAP50-95']))

    out_json = args.out + '_perclass.json'
    json.dump({'all': allrow, 'classes': classes},
              open(out_json, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('->', out_json)

    for fam in ('Microsoft YaHei', 'SimHei', 'DengXian'):
        try:
            font_manager.findfont(font_manager.FontProperties(family=fam),
                                  fallback_to_default=False)
            plt.rcParams['font.family'] = fam
            break
        except Exception:
            continue
    plt.rcParams['axes.unicode_minus'] = False

    classes = sorted(classes, key=lambda x: x['mAP50'])
    names = [c['class'] for c in classes]
    m50 = [c['mAP50'] for c in classes]
    m5095 = [c['mAP50-95'] for c in classes]
    inst = [c['instances'] for c in classes]

    fig, ax = plt.subplots(figsize=(11.5, 8.2))
    y = range(len(names))
    ax.barh([i + 0.20 for i in y], m50, height=0.38, color='#2b6cb0', label='mAP@0.5')
    ax.barh([i - 0.20 for i in y], m5095, height=0.38, color='#f6ad55', label='mAP@0.5:0.95')
    for i, (a, b, n) in enumerate(zip(m50, m5095, inst)):
        ax.text(a + 0.006, i + 0.20, '%.3f  (n=%d)' % (a, n), va='center', fontsize=8, color='#2b6cb0')
        ax.text(b + 0.006, i - 0.20, '%.3f' % b, va='center', fontsize=8, color='#b7791f')
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('AP')
    ax.set_xlim(0, 1.0)
    ax.grid(axis='x', ls=':', alpha=0.5)
    ax.legend(loc='lower right')
    t = args.title
    if allrow:
        t += '\n总体 mAP50 %.3f / mAP50-95 %.3f，验证集 %d 图 / %d 框' % (
            allrow['mAP50'], allrow['mAP50-95'], allrow['images'], allrow['instances'])
    ax.set_title(t, fontsize=11)
    fig.tight_layout()
    png = args.out + '_perclass.png'
    fig.savefig(png, dpi=120)
    print('->', png)


if __name__ == '__main__':
    main()
