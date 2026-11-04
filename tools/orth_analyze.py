# -*- coding: utf-8 -*-
"""拍屏正交实验分析：距离(近/远) × 视角(正对/偏斜) × 光照(正常/昏暗)，2×2×2 = 8 组。

每组一次「按下」，板端自动拍 3 对（ORT_<组>_<对>_a/b.jpg + 同名 .json）。
本脚本对每一对：
  1) a→b 相位相关求位移（板子稳定性检查）
  2) a 图 ↔ 源图 SIFT+RANSAC 单应性；由单应性算出
       - coverage：照片覆盖源图面积的百分比（= 距离的客观度量，不看用户描述）
       - trap_lr / trap_tb：投影四边形左右、上下边长比（= 倾斜的客观度量，正对≈1）
       - tgt_px：视野内目标的中位边长（px），换算成 512 输入后的等效像素
  3) 把 GT 投影到照片坐标系，与板端框 IoU>=0.5 贪心匹配，得 P/R/F1

用法：python tools\orth_analyze.py
"""
import csv
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e2e_auto_gt import load_gt, match, project_gt  # noqa: E402

ORTH_DIR = r'D:\rs-yolo\tmp\orth'
SRC_IMG = r'D:\rs-yolo\datasets\sodaa_ship_hbb\images\val\00063__800__0___0.jpg'
SRC_LBL = r'D:\rs-yolo\datasets\sodaa_ship_hbb\labels\val\00063__800__0___0.txt'
OUT = r'D:\rs-yolo\runs\orth'
MODEL_IN = 512

# 操作卡顺序（固定）：会话号 -> (距离, 视角, 光照)
PLAN = {
    1: ('近', '正对', '正常'),
    2: ('近', '正对', '昏暗'),
    3: ('近', '偏斜', '昏暗'),   # 用户实际先拍了第 4 组，故 3/4 对调
    4: ('近', '偏斜', '正常'),
    5: ('远', '正对', '正常'),
    6: ('远', '正对', '昏暗'),
    7: ('远', '偏斜', '正常'),
    8: ('远', '偏斜', '昏暗'),
}


def quad_metrics(H, src_w, src_h, ph_w, ph_h):
    """照片四角投影到源图，算覆盖率与四边形边长比。"""
    corners = np.array([[[0, 0], [ph_w, 0], [ph_w, ph_h], [0, ph_h]]], dtype=np.float32)
    q = cv2.perspectiveTransform(corners, H)[0]          # 照片角 -> 源图坐标
    # 面积（Shoelace），再裁到源图范围内
    x, y = q[:, 0], q[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    x0, y0 = max(x.min(), 0), max(y.min(), 0)
    x1, y1 = min(x.max(), src_w), min(y.max(), src_h)
    inter = max((x1 - x0) * (y1 - y0), 0.0)
    cover = inter / (src_w * src_h) * 100.0
    # 边长：0-1 上边、2-3 下边、1-2 右边、3-0 左边
    def d(i, j):
        return float(np.hypot(q[i][0] - q[j][0], q[i][1] - q[j][1]))
    top, right, bottom, left = d(0, 1), d(1, 2), d(2, 3), d(3, 0)
    trap_lr = (left / right) if right > 1e-6 else 0.0
    trap_tb = (top / bottom) if bottom > 1e-6 else 0.0
    return cover, trap_lr, trap_tb, (top + bottom + left + right) / 4.0


def target_px(gt_proj):
    if len(gt_proj) == 0:
        return 0.0
    w = gt_proj[:, 2] - gt_proj[:, 0]
    h = gt_proj[:, 3] - gt_proj[:, 1]
    s = np.sqrt(np.maximum(w * h, 0))
    s = s[s > 0]
    return float(np.median(s)) if len(s) else 0.0


def stability(pa, pb):
    ga = cv2.cvtColor(cv2.imread(pa), cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(cv2.imread(pb), cv2.COLOR_BGR2GRAY)
    (dx, dy), resp = cv2.phaseCorrelate(np.float32(ga), np.float32(gb))
    return float(np.hypot(dx, dy)), float(resp)


def main():
    os.makedirs(OUT, exist_ok=True)
    src, gts = load_gt(SRC_IMG, SRC_LBL)
    src_h, src_w = src.shape[:2]
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=8000)
    kp1, des1 = sift.detectAndCompute(src_gray, None)
    bf = cv2.BFMatcher()

    timing = {}
    tpath = os.path.join(ORTH_DIR, 'orth_timing.csv')
    if os.path.exists(tpath):
        with open(tpath) as f:
            for row in csv.reader(f):
                if len(row) >= 3:
                    timing[(int(row[0]), int(row[1]))] = int(row[2])

    per_group = {}
    for jp in sorted(glob.glob(os.path.join(ORTH_DIR, 'ORT_*.json'))):
        tag = os.path.splitext(os.path.basename(jp))[0]        # ORT_01_02
        parts = tag.split('_')
        sess, pair = int(parts[1]), int(parts[2])
        ap, bp = os.path.join(ORTH_DIR, tag + '_a.jpg'), os.path.join(ORTH_DIR, tag + '_b.jpg')
        if not (os.path.exists(ap) and os.path.exists(bp)):
            continue
        with open(jp, encoding='utf-8') as f:
            j = json.load(f)
        preds = np.array([[d['x'], d['y'], d['x'] + d['w'], d['y'] + d['h'], d['conf']]
                          for d in j['detections']], dtype=np.float32)
        if len(preds) == 0:
            print('%s 没有检测框，跳过' % tag)
            continue

        ph = cv2.imread(ap)
        ph_h, ph_w = ph.shape[:2]
        ph_gray = cv2.cvtColor(ph, cv2.COLOR_BGR2GRAY)
        kp2, des2 = sift.detectAndCompute(ph_gray, None)
        if des2 is None:
            print('%s 照片找不到特征' % tag)
            continue
        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) < 12:
            print('%s SIFT 匹配不足 (%d)' % (tag, len(good)))
            continue
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        inl = int(mask.sum()) if mask is not None else 0
        cover, tlr, ttb, side = quad_metrics(H, src_w, src_h, ph_w, ph_h)

        gt_proj = project_gt(gts, np.linalg.inv(H), ph_w, ph_h)
        tp, fp, fn, ious = match(preds, gt_proj)
        jitter, resp = stability(ap, bp)
        bright = float(cv2.cvtColor(ph, cv2.COLOR_BGR2GRAY).mean())
        tpx = target_px(gt_proj)

        row = {'tag': tag, 'sess': sess, 'pair': pair, 'inliers': inl,
               'cover_pct': round(cover, 2), 'trap_lr': round(tlr, 4),
               'trap_tb': round(ttb, 4), 'side_px': round(side, 1),
               'brightness': round(bright, 1), 'jitter_px': round(jitter, 3),
               'jitter_resp': round(resp, 4),
               'a_to_b_ms': timing.get((sess, pair), -1),
               'gt_in_view': int(len(gt_proj)), 'pred': int(len(preds)),
               'tp': tp, 'fp': fp, 'fn': fn,
               'precision': round(tp / max(tp + fp, 1), 4),
               'recall': round(tp / max(tp + fn, 1), 4),
               'mean_iou': round(float(np.mean(ious)) if ious else 0.0, 4),
               'tgt_px': round(tpx, 1),
               'tgt_px_in_model': round(tpx * MODEL_IN / max(ph_w, ph_h), 1)}
        per_group.setdefault(sess, []).append(row)
        print('%s cov=%5.1f%% trap=%.3f/%.3f bright=%5.1f jit=%.2fpx '
              'gt=%3d pred=%3d TP=%3d FP=%2d FN=%2d P=%.3f R=%.3f tgt=%.1fpx'
              % (tag, cover, tlr, ttb, bright, jitter, len(gt_proj), len(preds),
                 tp, fp, fn, row['precision'], row['recall'], tpx))

    summary = []
    for sess in sorted(per_group):
        rows = per_group[sess]
        tp = sum(r['tp'] for r in rows)
        fp = sum(r['fp'] for r in rows)
        fn = sum(r['fn'] for r in rows)
        P = tp / max(tp + fp, 1)
        R = tp / max(tp + fn, 1)
        dist, ang, light = PLAN.get(sess, ('?', '?', '?'))
        summary.append({
            'sess': sess, 'dist': dist, 'angle': ang, 'light': light,
            'n_pairs': len(rows),
            'cover_pct': round(float(np.mean([r['cover_pct'] for r in rows])), 2),
            'trap_lr': round(float(np.mean([r['trap_lr'] for r in rows])), 4),
            'brightness': round(float(np.mean([r['brightness'] for r in rows])), 1),
            'jitter_px': round(float(np.mean([r['jitter_px'] for r in rows])), 3),
            'gt_in_view': sum(r['gt_in_view'] for r in rows),
            'pred': sum(r['pred'] for r in rows),
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': round(P, 4), 'recall': round(R, 4),
            'f1': round(2 * P * R / max(P + R, 1e-9), 4),
            'mean_iou': round(float(np.mean([r['mean_iou'] for r in rows if r['mean_iou']])), 4),
            'tgt_px': round(float(np.mean([r['tgt_px'] for r in rows])), 1),
            'tgt_px_in_model': round(float(np.mean([r['tgt_px_in_model'] for r in rows])), 1),
        })

    print('\n=== 按组汇总 ===')
    print('组 距离 视角 光照 | 覆盖% 倾斜比 亮度 抖动 | GT  框数 TP  FP FN    P     R    F1   目标px')
    for s in summary:
        print('%2d %s  %s %s | %5.1f %.3f %5.1f %.2f | %3d %4d %3d %3d %2d  %.3f %.3f %.3f %5.1f'
              % (s['sess'], s['dist'], s['angle'], s['light'], s['cover_pct'],
                 s['trap_lr'], s['brightness'], s['jitter_px'], s['gt_in_view'],
                 s['pred'], s['tp'], s['fp'], s['fn'], s['precision'],
                 s['recall'], s['f1'], s['tgt_px']))

    out = {'plan': {str(k): v for k, v in PLAN.items()},
           'groups': summary,
           'pairs': [r for sess in sorted(per_group) for r in per_group[sess]]}
    op = os.path.join(OUT, 'orth_eval.json')
    with open(op, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('\n-> %s' % op)


if __name__ == '__main__':
    main()
