# SPDX-FileCopyrightText: 2026 XiaoFenglistens
# SPDX-License-Identifier: MIT
"""低分辨率基准集 —— 自动筛片 + 标注迁移（PC 侧）

思路：**GT 不需要"每张对得齐"，只需要"每张各自算得准"。**
每张照片独立用 SIFT + 单应性把原图 GT 框投过来，所以手抖导致画面不一样完全没关系，
只有"拍糊了导致单应性算歪"才是真问题 —— 这正是要筛的东西。

筛成三档（不是简单的收/丢）：
  A 清晰档：清晰度 >= 组内 p75 的 0.45 且内点 >= 300  —— 干净基线用
  B 可用档：内点 >= 150、重投影 <= 3.5px、可见率 >= 0.4 —— 真实条件用
  X 剔除档：单应性不可靠，标注不可信

用法:
  python tools\\cap_filter.py --dir D:\\rs-yolo\\captures
  python tools\\cap_filter.py --dir D:\\rs-yolo\\captures --vis 6   # 抽 6 张画投影框过目
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.environ.get("K230_ROOT", r"D:\rs-yolo")
SCENES_JSON = os.path.join(ROOT, "bench_scenes", "scenes.json")

TIER_A, TIER_B, TIER_X = "A", "B", "X"


def load_gt(label_path):
    """YOLO 归一化 cx cy w h -> 数组。"""
    rows = []
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            for ln in f:
                parts = ln.split()
                if len(parts) < 5:
                    continue
                rows.append([float(x) for x in parts[1:5]])
    except OSError:
        pass
    return np.array(rows, dtype=np.float32).reshape(-1, 4)


def gt_to_xyxy(boxes, w, h):
    if len(boxes) == 0:
        return np.zeros((0, 4), np.float32)
    cx, cy, bw, bh = boxes[:, 0] * w, boxes[:, 1] * h, boxes[:, 2] * w, boxes[:, 3] * h
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1).astype(np.float32)


def project_boxes(boxes_xyxy, H_mat, pw, ph):
    """原图框四角投到照片, 取 AABB + 可见率。"""
    if len(boxes_xyxy) == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
    out, vis = [], []
    for x1, y1, x2, y2 in boxes_xyxy:
        pts = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.float32)
        proj = cv2.perspectiveTransform(pts, H_mat).reshape(-1, 2)
        px1, py1 = float(proj[:, 0].min()), float(proj[:, 1].min())
        px2, py2 = float(proj[:, 0].max()), float(proj[:, 1].max())
        area = max(1e-6, (px2 - px1) * (py2 - py1))
        ix1, iy1 = max(0.0, px1), max(0.0, py1)
        ix2, iy2 = min(float(pw), px2), min(float(ph), py2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        out.append([px1, py1, px2, py2])
        vis.append(inter / area)
    return np.array(out, np.float32), np.array(vis, np.float32)


def analyse(photo_path, ref_gray, kp_ref, des_ref, gt_xyxy):
    img = cv2.imread(photo_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    ph, pw = img.shape[:2]
    res = {
        "file": os.path.basename(photo_path),
        "photo_w": pw, "photo_h": ph,
        "lapvar": round(float(cv2.Laplacian(img, cv2.CV_64F).var()), 2),
        "n_gt": int(len(gt_xyxy)),
    }
    sift = cv2.SIFT_create(nfeatures=6000)
    kp, des = sift.detectAndCompute(img, None)
    res["kp"] = len(kp) if kp else 0
    if des is None or len(kp) < 20:
        res.update({"inliers": 0, "reproj": 999.0, "visible": 0.0, "n_visible": 0,
                    "boxes": [], "reason": "特征点太少"})
        return res

    raw = cv2.BFMatcher().knnMatch(des, des_ref, k=2)
    good = [m for m, n in (p for p in raw if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 15:
        res.update({"inliers": len(good), "reproj": 999.0, "visible": 0.0, "n_visible": 0,
                    "boxes": [], "reason": "匹配太少"})
        return res

    src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)     # 照片
    dst = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)  # 原图
    Hm, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if Hm is None or mask is None:
        res.update({"inliers": 0, "reproj": 999.0, "visible": 0.0, "n_visible": 0,
                    "boxes": [], "reason": "单应性失败"})
        return res

    mask = mask.ravel().astype(bool)
    inl = int(mask.sum())
    proj = cv2.perspectiveTransform(src[mask], Hm).reshape(-1, 2)
    err = float(np.mean(np.linalg.norm(proj - dst[mask].reshape(-1, 2), axis=1)))

    Hinv = np.linalg.inv(Hm)
    boxes_ph, vis = project_boxes(gt_xyxy, Hinv, pw, ph)
    keep = vis >= 0.4
    # 只留框中心也在画面内的，避免半截框污染评测
    if len(boxes_ph):
        cx = (boxes_ph[:, 0] + boxes_ph[:, 2]) * 0.5
        cy = (boxes_ph[:, 1] + boxes_ph[:, 3]) * 0.5
        inside = (cx >= 0) & (cx < pw) & (cy >= 0) & (cy < ph)
        keep = keep & inside

    res.update({
        "inliers": inl,
        "reproj": round(err, 3),
        "visible": round(float(keep.mean()) if len(keep) else 0.0, 3),
        "n_visible": int(keep.sum()),
        "boxes": [[round(float(v), 1) for v in b] for b in boxes_ph[keep]],
    })
    return res


def draw_overlay(photo_path, boxes, out_path, title):
    img = cv2.imread(photo_path)
    if img is None:
        return
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
    cv2.putText(img, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.imwrite(out_path, img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "captures"))
    ap.add_argument("--scenes", default=SCENES_JSON)
    ap.add_argument("--min-inliers", type=int, default=150, help="B 档下限")
    ap.add_argument("--min-inliers-a", type=int, default=300, help="A 档下限")
    ap.add_argument("--max-reproj", type=float, default=3.5)
    ap.add_argument("--min-visible", type=float, default=0.4)
    ap.add_argument("--rel-lap", type=float, default=0.45, help="A 档清晰度 = 组内 p75 x 该系数")
    ap.add_argument("--abs-lap", type=float, default=40.0, help="清晰度绝对下限")
    ap.add_argument("--json", default=None)
    ap.add_argument("--vis", type=int, default=0, help="抽样画投影框的图片数（每组）")
    ap.add_argument("--out", default=os.path.join(ROOT, "bench_scenes"))
    args = ap.parse_args()

    if not os.path.isdir(args.dir):
        raise SystemExit("找不到采集目录: %s" % args.dir)
    with open(args.scenes, "r", encoding="utf-8") as f:
        meta = json.load(f)

    scene_of_group = {g["group"]: (g["scene"], g["dist"]) for g in meta["groups"]}
    scene_info = {s["scene"]: s for s in meta["scenes"]}
    ref_cache = {}

    files = sorted(f for f in os.listdir(args.dir) if f.lower().endswith(".jpg"))
    if not files:
        raise SystemExit("目录里没有 jpg: %s" % args.dir)

    report = {"params": {k: v for k, v in vars(args).items()}, "groups": {}}

    for fn in files:
        try:
            grp = int(fn[1:3])
        except Exception:
            continue
        if grp not in scene_of_group:
            continue
        scene, dist = scene_of_group[grp]

        if scene not in ref_cache:
            si = scene_info[scene]
            ref = cv2.imread(si["src_image"], cv2.IMREAD_GRAYSCALE)
            kp_ref, des_ref = cv2.SIFT_create(nfeatures=8000).detectAndCompute(ref, None)
            gt_xy = gt_to_xyxy(load_gt(si["src_label"]), ref.shape[1], ref.shape[0])
            ref_cache[scene] = (ref, kp_ref, des_ref, gt_xy)
            print("[场景] %s <- %s (%d 框)" % (scene, os.path.basename(si["src_image"]), len(gt_xy)))
        ref_gray, kp_ref, des_ref, gt_xy = ref_cache[scene]

        res = analyse(os.path.join(args.dir, fn), ref_gray, kp_ref, des_ref, gt_xy)
        if res is None:
            print("[失败] 读不出:", fn)
            continue
        g = report["groups"].setdefault("g%02d" % grp,
                                        {"scene": scene, "dist": dist, "files": []})
        g["files"].append(res)

    # ---- 两阶段定档：先算组内清晰度 p75，再定 A/B/X
    for gid, g in report["groups"].items():
        laps = sorted(f["lapvar"] for f in g["files"])
        p75 = laps[int(0.75 * (len(laps) - 1))] if laps else 0.0
        lap_a = max(args.abs_lap, args.rel_lap * p75)
        g["lap_p75"] = round(p75, 1)
        g["lap_thr_a"] = round(lap_a, 1)
        for f in g["files"]:
            reliable = (f["inliers"] >= args.min_inliers
                        and f["reproj"] <= args.max_reproj
                        and f["visible"] >= args.min_visible
                        and f["lapvar"] >= args.abs_lap)   # 太糊的连 B 档都不给
            if not reliable:
                f["tier"] = TIER_X
                why = []
                if f["lapvar"] < args.abs_lap:
                    why.append("模糊")
                if f["inliers"] < args.min_inliers:
                    why.append("内点少")
                if f["reproj"] > args.max_reproj:
                    why.append("重投影大")
                if f["visible"] < args.min_visible:
                    why.append("可见率低")
                f["reason"] = ",".join(why) or f.get("reason", "unknown")
            elif f["lapvar"] >= lap_a and f["inliers"] >= args.min_inliers_a:
                f["tier"] = TIER_A
            else:
                f["tier"] = TIER_B

    # ---- 汇总
    print("\n%-14s %-4s %8s %6s %6s %6s %5s" % ("file", "档", "清晰度", "内点", "重投影", "可见率", "框数"))
    tot = {"A": 0, "B": 0, "X": 0}
    for gid in sorted(report["groups"]):
        g = report["groups"][gid]
        print("\n-- %s  %s/%s  (清晰度 p75 = %.0f, A 档阈值 %.0f)" % (
            gid, g["scene"], g["dist"], g["lap_p75"], g["lap_thr_a"]))
        for f in g["files"]:
            tot[f["tier"]] += 1
            print("%-14s %-4s %8.1f %6d %6.2f %6.2f %5d %s" % (
                f["file"], f["tier"], f["lapvar"], f["inliers"], f["reproj"],
                f["visible"], f["n_visible"], "" if f["tier"] != TIER_X else "X " + f.get("reason", "")))

    n_all = sum(tot.values())
    print("\n=== 汇总 ===")
    print("共 %d 张：A 档 %d、B 档 %d、剔除 %d（可用 %.0f%%）" % (
        n_all, tot["A"], tot["B"], tot["X"], 100.0 * (tot["A"] + tot["B"]) / max(1, n_all)))

    # ---- 抽样可视化（检查标注迁移是否正确，这一步很关键）
    if args.vis > 0:
        vdir = os.path.join(args.out, "vis")
        os.makedirs(vdir, exist_ok=True)
        import random
        rnd = random.Random(0)
        for gid in sorted(report["groups"]):
            g = report["groups"][gid]
            pool = [f for f in g["files"] if f["tier"] != TIER_X]
            for f in rnd.sample(pool, min(args.vis, len(pool))):
                out_png = os.path.join(vdir, "%s_%s" % (gid, f["file"][:-4] + ".jpg"))
                draw_overlay(os.path.join(args.dir, f["file"]), f["boxes"], out_png,
                             "%s %s  GT=%d  lap=%.0f" % (gid, f["tier"], f["n_visible"], f["lapvar"]))
        print("可视化 -> %s" % vdir)

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print("报告 -> %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
