# -*- coding: utf-8 -*-
"""正交实验二次汇总：剔除「按下瞬间」那一对，做三因素趋势 + 质量诊断。

输入 D:\\rs-yolo\\runs\\orth\\orth_eval.json
输出 D:\\rs-yolo\\runs\\orth\\orth_summary.json / orth_fit.json / orth_curve.png
"""
import json
import os

import numpy as np

RUN = r'D:\rs-yolo\runs\orth'
SRC = os.path.join(RUN, 'orth_eval.json')

# 只保留稳定对（b 对），a 对是按下瞬间仍在动的那一张
STABLE_PAIRS = (2, 3)


def load():
    return json.load(open(SRC, encoding='utf-8'))


def agg(pairs):
    gt = sum(p['gt_in_view'] for p in pairs)
    tp = sum(p['tp'] for p in pairs)
    fp = sum(p['fp'] for p in pairs)
    fn = sum(p['fn'] for p in pairs)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        'gt': gt, 'pred': sum(p['pred'] for p in pairs), 'tp': tp, 'fp': fp, 'fn': fn,
        'precision': round(prec, 4), 'recall': round(rec, 4), 'f1': round(f1, 4),
        'tgt_px': round(float(np.median([p['tgt_px'] for p in pairs])), 2),
        'brightness': round(float(np.median([p['brightness'] for p in pairs])), 1),
        'trap_lr': round(float(np.median([p['trap_lr'] for p in pairs])), 3),
        'trap_tb': round(float(np.median([p['trap_tb'] for p in pairs])), 3),
        'jitter_px': round(float(np.median([p['jitter_px'] for p in pairs])), 2),
        'mean_iou': round(float(np.median([p['mean_iou'] for p in pairs])), 3),
    }


def main():
    d = load()
    plan = d['plan']
    pairs = d['pairs']
    stable = [p for p in pairs if p['pair'] in STABLE_PAIRS]
    shaky = [p for p in pairs if p['pair'] not in STABLE_PAIRS]

    groups = {}
    for sid in sorted(plan, key=lambda x: int(x)):
        g = [p for p in stable if str(p['sess']) == sid]
        if not g:
            continue
        row = agg(g)
        row['sess'] = int(sid)
        row['dist'], row['view'], row['light'] = plan[sid]
        groups[sid] = row

    print('=== 稳定对（每组第 2、3 对合并，每组 GT=210）===')
    print('组 距离 视角 光照 | 目标px 亮度 倾斜比 抖动 | TP  FP  FN    P     R    F1')
    for sid in sorted(groups, key=int):
        r = groups[sid]
        print('%2d %-3s %-3s %-3s | %5.1f %5.1f %5.3f %5.2f | %3d %3d %3d  %.3f %.3f %.3f'
              % (r['sess'], r['dist'], r['view'], r['light'], r['tgt_px'], r['brightness'],
                 r['trap_lr'], r['jitter_px'], r['tp'], r['fp'], r['fn'],
                 r['precision'], r['recall'], r['f1']))

    if shaky:
        s = agg(shaky)
        print('\n=== 按下瞬间那一对（每组第 1 对，共 %d 张）===' % len(shaky))
        print('抖动中位 %.1f px（稳定对 %.1f px）  召回 %.3f（稳定对 %.3f）'
              % (s['jitter_px'],
                 float(np.median([p['jitter_px'] for p in stable])),
                 s['recall'], agg(stable)['recall']))

    print('\n=== 单因素趋势 ===')
    def line(key, label, bins):
        print('%-8s' % label, end='')
        for lo, hi, name in bins:
            v = [p['recall'] for p in stable if lo <= p[key] < hi]
            tp = sum(p['tp'] for p in stable if lo <= p[key] < hi)
            gt = sum(p['gt_in_view'] for p in stable if lo <= p[key] < hi)
            r = tp / gt if gt else 0.0
            print('  %s[%d图]=%.3f' % (name, len(v), r), end='')
        print()

    line('tgt_px', '目标像素', [(0, 9, '<9'), (9, 11, '9-11'), (11, 13, '11-13'), (13, 99, '>13')])
    line('trap_lr', '左右比', [(0, 0.7, '<0.7'), (0.7, 0.85, '0.7-0.85'), (0.85, 99, '>0.85')])
    line('brightness', '亮度', [(0, 60, '<60'), (60, 70, '60-70'), (70, 999, '>70')])

    out = {'plan': plan, 'stable_pairs': list(STABLE_PAIRS),
           'groups': groups,
           'press_moment': agg(shaky) if shaky else None,
           'stable_all': agg(stable)}
    json.dump(out, open(os.path.join(RUN, 'orth_summary.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('\n->', os.path.join(RUN, 'orth_summary.json'))


if __name__ == '__main__':
    main()
