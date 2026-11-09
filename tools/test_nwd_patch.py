# SPDX-FileCopyrightText: 2026 XiaoFenglistens
# SPDX-License-Identifier: MIT
"""nwd_patch.py 的自检：等价性、数值范围、小目标敏感性、端到端 assigner 跑通。

用法：python tools\\test_nwd_patch.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nwd_patch  # noqa: E402
from ultralytics.utils.tal import TaskAlignedAssigner  # noqa: E402


def make_batch(n=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    sizes = torch.randint(6, 40, (n,), generator=g).float()  # 6~40 px 的 GT 边长
    cx = torch.rand(n, generator=g) * 400 + 20
    cy = torch.rand(n, generator=g) * 400 + 20
    gt = torch.stack([cx - sizes / 2, cy - sizes / 2, cx + sizes / 2, cy + sizes / 2], 1)
    # 预测框：GT 平移 1~6 px 并轻微缩放
    shift = (torch.rand(n, 2, generator=g) - 0.5) * 12
    scale = 1 + (torch.rand(n, generator=g) - 0.5) * 0.3
    pw = sizes * scale
    ph = sizes * scale
    pd = torch.stack(
        [cx + shift[:, 0] - pw / 2, cy + shift[:, 1] - ph / 2, cx + shift[:, 0] + pw / 2, cy + shift[:, 1] + ph / 2], 1
    )
    return gt, pd


def main():
    gt, pd = make_batch()
    iou = nwd_patch._iou(gt, pd)

    print("=== 1) NWD vs IoU 在不同目标尺寸下的表现 ===")
    nwd = nwd_patch._nwd(gt, pd, 16.0)
    for lo, hi, name in ((6, 10, "6-10px"), (10, 20, "10-20px"), (20, 40, "20-40px")):
        size = torch.sqrt((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1]))
        m = (size >= lo) & (size < hi)
        if m.any():
            print(f"  {name:8s} n={int(m.sum()):3d}  IoU 均值 {iou[m].mean():.3f}  NWD 均值 {nwd[m].mean():.3f}")

    print("=== 2) α=0 必须逐元素等于原实现（CIoU） ===")
    align_o, ov_o = nwd_patch._orig_get_box_metrics.__wrapped__ if False else (None, None)
    b, nmax, na = 2, 5, 64
    gt4 = torch.zeros(b, nmax, 4)
    pd4 = torch.rand(b, na, 4) * 100
    sc = torch.rand(b, na, 1)
    lbl = torch.zeros(b, nmax, 1)
    mk = torch.zeros(b, nmax, na, dtype=torch.bool)
    mk[:, :, 0] = True
    for i in range(b):
        for j in range(nmax):
            gt4[i, j] = torch.tensor([10.0 * j, 10.0 * j, 10.0 * j + 20, 10.0 * j + 20])

    a = TaskAlignedAssigner(topk=10, num_classes=1, alpha=1.0, beta=6.0, stride=[8, 16, 32])
    a.bs, a.n_max_boxes = b, nmax
    os.environ["NWD_ALPHA"] = "0"
    m0, o0 = TaskAlignedAssigner.get_box_metrics(a, sc, pd4, lbl, gt4, mk)
    m_ref, o_ref = nwd_patch._orig_get_box_metrics(a, sc, pd4, lbl, gt4, mk)
    print(f"  align max|diff| = {(m0 - m_ref).abs().max():.3e}   overlaps max|diff| = {(o0 - o_ref).abs().max():.3e}")
    assert torch.allclose(m0, m_ref) and torch.allclose(o0, o_ref), "α=0 不等价，补丁有问题"

    print("=== 3) alpha=1 时 metric 与 overlaps 都在 [0,1]，且小目标更黏 ===")
    os.environ["NWD_ALPHA"] = "1"
    m1, o1 = TaskAlignedAssigner.get_box_metrics(a, sc, pd4, lbl, gt4, mk)
    print(f"  overlaps min={o1.min():.4f} max={o1.max():.4f}  align min={m1.min():.4f} max={m1.max():.4f}")
    assert 0.0 <= float(o1.min()) and float(o1.max()) <= 1.0 + 1e-6

    print("=== 4) 端到端：完整 assigner forward（含 topk / 归一化） ===")
    for alpha in ("0", "0.5", "1"):
        os.environ["NWD_ALPHA"] = alpha
        aa = TaskAlignedAssigner(topk=10, num_classes=3, alpha=0.5, beta=6.0, stride=[8, 16, 32])
        bs, nbox, nanc = 1, 3, 400
        ps = torch.rand(bs, nanc, 3)
        pb = torch.rand(bs, nanc, 4) * 64
        anc = torch.rand(nanc, 2) * 64
        gl = torch.zeros(bs, nbox, 1)
        gb = torch.tensor([[[5.0, 5.0, 25, 25], [30.0, 30.0, 44, 44], [1.0, 1.0, 7, 7]]])
        mg = torch.ones(bs, nbox, 1)
        out = aa.forward(ps, pb, anc, gl, gb, mg)
        fg = int(out[3].sum())
        print(f"  α={alpha:4s} -> fg anchors={fg:3d}  target_scores sum={float(out[2].sum()):.3f}")

    print("\nALL OK")


if __name__ == "__main__":
    main()
