# -*- coding: utf-8 -*-
"""对比板端 20 类检测与 PC 参照：数量、类别一致率、逐框 IoU。"""
import glob
import json
import os

import numpy as np

BOARD = r'D:\rs-yolo\tmp\dior_board'
PC = r'D:\rs-yolo\runs\dior20_pc_ref.json'
OUT = r'D:\rs-yolo\runs\dior20_board_check.json'


def iou(a, b):
    x1 = max(a['x'], b['x'])
    y1 = max(a['y'], b['y'])
    x2 = min(a['x'] + a['w'], b['x'] + b['w'])
    y2 = min(a['y'] + a['h'], b['y'] + b['h'])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = a['w'] * a['h'] + b['w'] * b['h'] - inter
    return inter / ua if ua > 0 else 0.0


def main():
    pc = json.load(open(PC, encoding='utf-8'))
    rows = []
    for p in sorted(glob.glob(os.path.join(BOARD, 'DET_d*.json'))):
        key = os.path.basename(p)[4:-5]           # d1
        bd = json.load(open(p, encoding='utf-8'))
        bdets = bd['detections']
        pdets = pc[key]['detections']
        used = set()
        ious = []
        cls_ok = 0
        for b in bdets:
            best, bi = 0.0, -1
            for i, q in enumerate(pdets):
                if i in used:
                    continue
                v = iou(b, q)
                if v > best:
                    best, bi = v, i
            if bi >= 0 and best >= 0.3:
                used.add(bi)
                ious.append(best)
                if b['cls'] == pdets[bi]['cls']:
                    cls_ok += 1
        rows.append({
            'image': key, 'name': pc[key]['name'],
            'board': len(bdets), 'pc': len(pdets),
            'matched_iou30': len(ious),
            'iou_median': round(float(np.median(ious)), 3) if ious else 0.0,
            'iou_p90': round(float(np.percentile(ious, 90)), 3) if ious else 0.0,
            'iou_gt50': sum(1 for v in ious if v >= 0.5),
            'cls_agree': cls_ok,
            'cls_agree_rate': round(cls_ok / len(ious), 3) if ious else 0.0,
        })
        print('%-4s %-8s board %3d / pc %3d | 配对 %3d  IoU中位 %.3f  >0.5 %2d  同类 %.0f%%'
              % (key, pc[key]['name'], len(bdets), len(pdets), len(ious),
                 rows[-1]['iou_median'], rows[-1]['iou_gt50'], rows[-1]['cls_agree_rate'] * 100))

    tb = sum(r['board'] for r in rows)
    tp = sum(r['pc'] for r in rows)
    tm = sum(r['matched_iou30'] for r in rows)
    t50 = sum(r['iou_gt50'] for r in rows)
    tc = sum(r['cls_agree'] for r in rows)
    alliou = []
    print('\n合计：板端 %d 框 / PC %d 框；配对 %d（IoU>=0.3），其中 IoU>=0.5 的 %d（%.1f%%）；类别一致 %d（%.1f%%）'
          % (tb, tp, tm, t50, 100.0 * t50 / max(1, tm), tc, 100.0 * tc / max(1, tm)))
    json.dump({'rows': rows, 'total': {'board': tb, 'pc': tp, 'matched': tm,
                                       'iou_gt50': t50, 'cls_agree': tc}}, open(OUT, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('->', OUT)


if __name__ == '__main__':
    main()
