# SPDX-FileCopyrightText: 2026 XiaoFenglistens
# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""DIOR (COCO 标注) -> YOLO 多类数据集。

用法：
    python tools\dior_to_yolo.py --build        # 建 D:\rs-yolo\datasets\dior20
    python tools\dior_to_yolo.py --check        # 只统计不写盘

输入：
    D:\rs-yolo\datasets\crossdomain\dior\labels\COCO_labels\{train,val}.json
    D:\rs-yolo\datasets\crossdomain\dior\val\val\images\*.jpg        （val 图）
    D:\rs-yolo\datasets\crossdomain\dior\train\train\images\*.jpg    （train 图，train.zip 解压后）

输出：
    D:\rs-yolo\datasets\dior20\images\{train,val}\*.jpg
    D:\rs-yolo\datasets\dior20\labels\{train,val}\*.txt
    D:\rs-yolo\datasets\dior20\data.yaml
"""
import argparse
import json
import os
import shutil
import sys

DIOR = r'D:\rs-yolo\datasets\crossdomain\dior'
COCO = os.path.join(DIOR, 'labels', 'COCO_labels')
OUT = r'D:\rs-yolo\datasets\dior20'

# 图像目录候选（zip 内自带一层同名目录，两种布局都试）
IMG_DIRS = {
    'train': [os.path.join(DIOR, 'train', 'train', 'images'),
              os.path.join(DIOR, 'train', 'images')],
    'val': [os.path.join(DIOR, 'val', 'val', 'images'),
            os.path.join(DIOR, 'val', 'images')],
}

# 20 个前景类（COCO 的 category_id 1..20；id 0 是 background，丢弃）
NAMES = [
    'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
    'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station',
    'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium',
    'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill',
]


def find_img_dir(split):
    for d in IMG_DIRS[split]:
        if os.path.isdir(d):
            return d
    return None


def load_split(split):
    p = os.path.join(COCO, '%s.json' % split)
    d = json.load(open(p, encoding='utf-8'))
    cats = {c['id']: c['name'] for c in d['categories']}
    imgs = {im['id']: im for im in d['images']}
    return d, cats, imgs


def convert(split, img_dir, dst_img, dst_lbl, copy_images=True):
    d, cats, imgs = load_split(split)
    by_img = {}
    skipped_crowd = skipped_bad = 0
    for a in d['annotations']:
        if a.get('iscrowd'):
            skipped_crowd += 1
            continue
        cid = a['category_id']
        if cid == 0:                       # background，跳过
            continue
        im = imgs[a['image_id']]
        W, H = im['width'], im['height']
        x, y, w, h = a['bbox']
        if w <= 1 or h <= 1:
            skipped_bad += 1
            continue
        cx = (x + w / 2.0) / W
        cy = (y + h / 2.0) / H
        by_img.setdefault(a['image_id'], []).append(
            '%d %.6f %.6f %.6f %.6f' % (cid - 1, cx, cy, w / W, h / H))

    for dd in (dst_img, dst_lbl):
        os.makedirs(dd, exist_ok=True)

    n_img = n_box = n_miss = 0
    for iid, im in sorted(imgs.items()):
        name = im['file_name']
        src = os.path.join(img_dir, name)
        if not os.path.exists(src):
            n_miss += 1
            continue
        if copy_images:
            shutil.copy2(src, os.path.join(dst_img, name))
        stem = os.path.splitext(name)[0]
        lines = by_img.get(iid, [])
        with open(os.path.join(dst_lbl, stem + '.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
            if lines:
                f.write('\n')
        n_img += 1
        n_box += len(lines)
    return {
        'split': split, 'images': n_img, 'boxes': n_box,
        'missing_files': n_miss, 'skip_crowd': skipped_crowd, 'skip_bad_box': skipped_bad,
    }


def write_yaml():
    p = os.path.join(OUT, 'data.yaml')
    lines = [
        '# DIOR 20 类（COCO 标注转 YOLO），images 均为 800x800',
        'path: %s' % OUT.replace('\\', '/'),
        'train: images/train',
        'val: images/val',
        '',
        'nc: %d' % len(NAMES),
        'names:',
    ]
    for i, n in enumerate(NAMES):
        lines.append("  %d: '%s'" % (i, n))
    open(p, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('data.yaml ->', p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', action='store_true', help='真正写盘')
    ap.add_argument('--check', action='store_true', help='只统计')
    ap.add_argument('--splits', default='train,val')
    args = ap.parse_args()
    if not (args.build or args.check):
        ap.error('要么 --build 要么 --check')

    stats = []
    for split in args.splits.split(','):
        split = split.strip()
        img_dir = find_img_dir(split)
        if img_dir is None:
            print('[%s] 图像目录不存在，跳过（train.zip 还没解压？）' % split)
            continue
        n_src = len([f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')])
        print('[%s] 图像目录 %s（%d 张 jpg）' % (split, img_dir, n_src))
        if args.check:
            d, cats, imgs = load_split(split)
            print('      json: %d 图 / %d 框' % (len(imgs), len(d['annotations'])))
            continue
        st = convert(split, img_dir,
                     os.path.join(OUT, 'images', split),
                     os.path.join(OUT, 'labels', split))
        stats.append(st)
        print('      -> %d 图 / %d 框，缺图 %d，跳过 crowd %d / 坏框 %d'
              % (st['images'], st['boxes'], st['missing_files'],
                 st['skip_crowd'], st['skip_bad_box']))

    if args.build and stats:
        write_yaml()
        # 类别分布
        import collections
        c = collections.Counter()
        for split in args.splits.split(','):
            ld = os.path.join(OUT, 'labels', split.strip())
            if not os.path.isdir(ld):
                continue
            for f in os.listdir(ld):
                for line in open(os.path.join(ld, f), encoding='utf-8'):
                    if line.strip():
                        c[int(line.split()[0])] += 1
        print('\n类别分布（全部 split 合计）：')
        for i, n in enumerate(NAMES):
            print('  %2d %-26s %6d' % (i, n, c.get(i, 0)))


if __name__ == '__main__':
    main()
