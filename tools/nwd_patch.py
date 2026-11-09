# SPDX-FileCopyrightText: 2026 XiaoFenglistens
# SPDX-License-Identifier: MIT
"""NWD（Normalized Wasserstein Distance）小目标标签分配补丁。

用法：在任何 ultralytics 训练脚本之前 `import nwd_patch`，由环境变量控制：
    NWD_ALPHA  0.0 = 纯 IoU（基线）；1.0 = 纯 NWD；0<a<1 = 混合
    NWD_C      固定归一化常数（像素）。0 或未设 = 按 batch 内 GT 尺寸自适应
    NWD_MODE   blend（默认，全局 α）| adaptive（α 随 GT 尺寸变化）

为什么改这里：TAL 用 IoU 做正样本选择。对 8~16 px 的小目标，中心偏 3 px 就
IoU<0.5，正样本又少又抖；NWD 把框建模成二维高斯，用它们之间的 Wasserstein
距离衡量相似度，对位置偏移是"连续衰减"而不是"断崖式归零"。
改的是训练侧分配器，推理图与板端完全不变（零帧率成本）。
"""

import os
import math

import torch

from ultralytics.utils.tal import TaskAlignedAssigner


def _nwd(gt_bboxes, pd_bboxes, c):
    """Normalized Wasserstein Distance 相似度，输入 xyxy，返回 (N,) 的 [0,1]。"""
    g_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) * 0.5
    g_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) * 0.5
    g_w = (gt_bboxes[:, 2] - gt_bboxes[:, 0]).abs() * 0.5
    g_h = (gt_bboxes[:, 3] - gt_bboxes[:, 1]).abs() * 0.5
    p_cx = (pd_bboxes[:, 0] + pd_bboxes[:, 2]) * 0.5
    p_cy = (pd_bboxes[:, 1] + pd_bboxes[:, 3]) * 0.5
    p_w = (pd_bboxes[:, 2] - pd_bboxes[:, 0]).abs() * 0.5
    p_h = (pd_bboxes[:, 3] - pd_bboxes[:, 1]).abs() * 0.5

    # 二维高斯之间的 2-Wasserstein 距离平方
    d2 = (g_cx - p_cx) ** 2 + (g_cy - p_cy) ** 2 + (g_w - p_w) ** 2 + (g_h - p_h) ** 2
    return torch.exp(-torch.sqrt(d2.clamp_min(0)) / c)


def _iou(gt_bboxes, pd_bboxes):
    """普通 IoU（不带 CIoU 惩罚项），逐对计算，用于和 NWD 混合。"""
    ix1 = torch.max(gt_bboxes[:, 0], pd_bboxes[:, 0])
    iy1 = torch.max(gt_bboxes[:, 1], pd_bboxes[:, 1])
    ix2 = torch.min(gt_bboxes[:, 2], pd_bboxes[:, 2])
    iy2 = torch.min(gt_bboxes[:, 3], pd_bboxes[:, 3])
    inter = (ix2 - ix1).clamp_min(0) * (iy2 - iy1).clamp_min(0)
    area_g = (gt_bboxes[:, 2] - gt_bboxes[:, 0]).clamp_min(0) * (gt_bboxes[:, 3] - gt_bboxes[:, 1]).clamp_min(0)
    area_p = (pd_bboxes[:, 2] - pd_bboxes[:, 0]).clamp_min(0) * (pd_bboxes[:, 3] - pd_bboxes[:, 1]).clamp_min(0)
    return (inter / (area_g + area_p - inter).clamp_min(1e-9)).clamp(0, 1)


def _alpha_for(gt_boxes, alpha, mode):
    """按 GT 尺寸给出每个框的混合权重 α。

    adaptive 模式：目标越小 α 越大（越小越依赖 NWD），
    以 C 为参考：边长 <= C/2 用满 α，边长 >= 2C 退回纯 IoU。
    """
    if mode == "gate":
        # 硬门控：只有小目标用 NWD，大目标完全退回 IoU（避免大目标被稀释）
        gate = max(float(os.environ.get("NWD_GATE_PX", "20") or 20), 1.0)
        size = torch.sqrt(
            (gt_boxes[:, 2] - gt_boxes[:, 0]).abs().clamp_min(1)
            * (gt_boxes[:, 3] - gt_boxes[:, 1]).abs().clamp_min(1)
        )
        return torch.where(size < gate, float(alpha), 0.0).to(gt_boxes.dtype)
    if mode != "adaptive":
        return torch.full((gt_boxes.shape[0],), float(alpha), device=gt_boxes.device, dtype=gt_boxes.dtype)
    size = torch.sqrt(
        (gt_boxes[:, 2] - gt_boxes[:, 0]).abs().clamp_min(1) * (gt_boxes[:, 3] - gt_boxes[:, 1]).abs().clamp_min(1)
    )
    ref = max(float(os.environ.get("NWD_ADAPT_REF", 24.0)), 1.0)
    # size=ref/4 → 满 α；size=ref → α/2；size>=2ref → 0
    w = ((2.0 * ref - size) / (1.75 * ref)).clamp(0, 1)
    return (w * float(alpha)).to(gt_boxes.dtype)


_orig_get_box_metrics = TaskAlignedAssigner.get_box_metrics


def get_box_metrics_nwd(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
    alpha = float(os.environ.get("NWD_ALPHA", "0") or 0)
    if alpha <= 0.0:
        return _orig_get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt)

    mode = (os.environ.get("NWD_MODE", "blend") or "blend").lower()
    na = pd_bboxes.shape[-2]
    mask_gt = mask_gt.bool()
    shape = self.bs, self.n_max_boxes, na
    indices = mask_gt.nonzero(as_tuple=True)
    if indices[0].numel() == 0:
        return _orig_get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt)

    bbox_scores = pd_scores[indices[0], indices[2], gt_labels[indices[0], indices[1], 0].long()]
    gt_b = gt_bboxes[indices[:2]]
    pd_b = pd_bboxes[indices[0], indices[2]]

    iou_v = _iou(gt_b, pd_b)
    c_fixed = float(os.environ.get("NWD_C", "0") or 0)
    if c_fixed > 0:
        c_val = c_fixed
    else:
        # 自适应：用本 batch 有效 GT 的平均边长，夹在 [6, 40]
        size = torch.sqrt(
            (gt_b[:, 2] - gt_b[:, 0]).abs().clamp_min(1) * (gt_b[:, 3] - gt_b[:, 1]).abs().clamp_min(1)
        )
        c_val = float(size.mean().item())
        c_val = min(max(c_val, 6.0), 40.0)
    nwd_v = _nwd(gt_b, pd_b, c_val)

    a_vec = _alpha_for(gt_b, alpha, mode)
    sim = a_vec * nwd_v + (1.0 - a_vec) * iou_v

    align_values = bbox_scores.pow(self.alpha) * sim.pow(self.beta)
    overlaps = torch.zeros(shape, dtype=pd_bboxes.dtype, device=pd_bboxes.device)
    align_metric = torch.zeros(shape, dtype=align_values.dtype, device=pd_scores.device)
    if mode == "rank":
        # 只在"排序"（top-k 选正样本）时用 NWD，归一化与多 GT 仲裁仍用原始 CIoU：
        # 保住定位紧致度，只提高小目标被选中的机会。
        align_metric[indices] = align_values
        overlaps[indices] = self.iou_calculation(gt_b, pd_b)
    else:
        overlaps[indices] = sim
        align_metric[indices] = align_values
    return align_metric, overlaps


_patched = False


def apply():
    """把 NWD 版 get_box_metrics 挂到 TaskAlignedAssigner 上（幂等）。"""
    global _patched
    if _patched:
        return config()
    TaskAlignedAssigner.get_box_metrics = get_box_metrics_nwd
    _patched = True
    cfg = config()
    print(f"[NWD] patch applied: {cfg}")
    return cfg


def config():
    return {
        "alpha": float(os.environ.get("NWD_ALPHA", "0") or 0),
        "mode": (os.environ.get("NWD_MODE", "blend") or "blend").lower(),
        "C": float(os.environ.get("NWD_C", "0") or 0),
        "adapt_ref": float(os.environ.get("NWD_ADAPT_REF", "24") or 24),
    }


# import 即生效，方便 `python -c "import nwd_patch"` 直接验证
if os.environ.get("NWD_ALPHA") is not None:
    apply()
