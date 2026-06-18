"""
utils/eval_custom.py — Evaluation functions cho Custom 2D Dataset
=================================================================
Hỗ trợ:
  - Binary segmentation (num_classes=2)
  - Multi-class segmentation (num_classes>2)

Metrics:
  - Precision (per-class + mean)
  - Recall    (per-class + mean)
  - IoU / Jaccard (per-class + mIoU)
  - Dice Score (per-class + mean)
  - HD95 (Hausdorff Distance 95th percentile) — nếu medpy được cài

Cách dùng từ test_custom.py:
    from utils.eval_custom import test_model_custom
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from medpy.metric.binary import hd95, dc
    _HAS_MEDPY = True
except ImportError:
    _HAS_MEDPY = False
    print("[eval_custom] ⚠️  medpy chưa được cài (pip install medpy). "
          "HD95 sẽ bị bỏ qua, chỉ tính Dice.")


# ---------------------------------------------------------------------------
# Confusion-matrix based metrics: Precision, Recall, IoU
# ---------------------------------------------------------------------------
def compute_segmentation_metrics(pred: np.ndarray, target: np.ndarray,
                                  num_classes: int = 2,
                                  ignore_background: bool = True):
    """
    Tính Precision, Recall, IoU cho từng class dựa trên confusion matrix.

    Args:
        pred             : numpy array 2D (H, W), giá trị nguyên [0, num_classes)
        target           : numpy array 2D (H, W), giá trị nguyên [0, num_classes)
        num_classes      : số class (bao gồm background class 0)
        ignore_background: bỏ qua class 0 khi tính mean

    Returns:
        precision_list : list[float] — Precision per class (bỏ background nếu flag=True)
        recall_list    : list[float] — Recall per class
        iou_list       : list[float] — IoU per class
        mean_precision : float
        mean_recall    : float
        mean_iou       : float
    """
    class_range = range(1, num_classes) if ignore_background else range(num_classes)
    precision_list, recall_list, iou_list = [], [], []

    for c in class_range:
        pred_c   = (pred == c)
        target_c = (target == c)

        tp = float((pred_c & target_c).sum())
        fp = float((pred_c & ~target_c).sum())
        fn = float((~pred_c & target_c).sum())

        precision = tp / (tp + fp + 1e-8) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else 0.0
        iou       = tp / (tp + fp + fn + 1e-8) if (tp + fp + fn) > 0 else 0.0

        # Nếu GT và pred đều rỗng → perfect score
        if target_c.sum() == 0 and pred_c.sum() == 0:
            precision, recall, iou = 1.0, 1.0, 1.0

        precision_list.append(precision)
        recall_list.append(recall)
        iou_list.append(iou)

    mean_precision = float(np.mean(precision_list)) if precision_list else 0.0
    mean_recall    = float(np.mean(recall_list))    if recall_list    else 0.0
    mean_iou       = float(np.mean(iou_list))       if iou_list       else 0.0

    return precision_list, recall_list, iou_list, mean_precision, mean_recall, mean_iou


# ---------------------------------------------------------------------------
# Utility: tính Dice và HD95 cho MỘT cặp pred/target (numpy arrays 2D/3D)
# ---------------------------------------------------------------------------
def compute_metrics_per_class(pred: np.ndarray, target: np.ndarray,
                               num_classes: int = 2,
                               ignore_background: bool = True):
    """
    Tính Dice (và HD95 nếu medpy khả dụng) + Precision/Recall/IoU cho từng class.

    Returns:
        dice_scores      : list[float]
        hd95_scores      : list[float] hoặc None nếu medpy không có
        precision_list   : list[float]
        recall_list      : list[float]
        iou_list         : list[float]
    """
    class_range = range(1, num_classes) if ignore_background else range(num_classes)
    dice_scores, hd95_scores = [], []

    for c in class_range:
        pred_c   = (pred == c).astype(np.uint8)
        target_c = (target == c).astype(np.uint8)

        # Cả hai đều rỗng → hoàn hảo
        if target_c.sum() == 0 and pred_c.sum() == 0:
            dice_scores.append(1.0)
            hd95_scores.append(0.0)
            continue
        # Một bên rỗng → sai hoàn toàn
        if target_c.sum() == 0 or pred_c.sum() == 0:
            dice_scores.append(0.0)
            hd95_scores.append(100.0)
            continue

        dice_scores.append(dc(pred_c, target_c) if _HAS_MEDPY else _dice_numpy(pred_c, target_c))
        if _HAS_MEDPY:
            try:
                hd95_scores.append(hd95(pred_c, target_c))
            except Exception:
                hd95_scores.append(100.0)
        else:
            hd95_scores.append(float("nan"))

    # Tính P / R / IoU
    precision_list, recall_list, iou_list, _, _, _ = compute_segmentation_metrics(
        pred, target, num_classes=num_classes, ignore_background=ignore_background
    )

    return dice_scores, hd95_scores if _HAS_MEDPY else None, precision_list, recall_list, iou_list


def _dice_numpy(pred_c: np.ndarray, target_c: np.ndarray, smooth: float = 1e-5) -> float:
    """Dice thuần numpy (backup khi không có medpy)."""
    inter = (pred_c * target_c).sum()
    return float(2.0 * inter + smooth) / float(pred_c.sum() + target_c.sum() + smooth)


# ---------------------------------------------------------------------------
# Hàm inference + evaluation trên toàn bộ tập test (2D images)
# ---------------------------------------------------------------------------
def test_model_custom(
    model,
    data_root: str,
    img_transform,
    split: str = "test",
    num_classes: int = 2,
    img_size: int = 512,
    device: str = "cuda",
    save_file: str = "results_custom.txt",
    ignore_background: bool = True,
    mask_suffix: str = "",
    list_file: str = None,
    images_dir: str = "images",
    labels_dir: str = "labels",
):
    """
    Chạy inference và đánh giá model trên tập 2D images tùy chỉnh.

    Args:
        model         : Mô hình PyTorch đã load weight
        data_root     : Thư mục gốc chứa images/, labels/
        img_transform : Transform áp dụng cho ảnh đầu vào
        split         : 'train' | 'val' | 'test'
        num_classes   : Số class phân đoạn (bao gồm background)
        img_size      : Kích thước resize
        device        : 'cuda' hoặc 'cpu'
        save_file     : Đường dẫn lưu kết quả text
        ignore_background : Bỏ qua class 0 khi tính metrics
        mask_suffix   : Suffix thêm vào tên file mask
        list_file     : Tên file danh sách (None → tự scan thư mục)
        images_dir    : Tên thư mục ảnh (mặc định 'images')
        labels_dir    : Tên thư mục mask (mặc định 'labels')

    Returns:
        results : list of (stem, dice_list, hd95_list, precision_list, recall_list, iou_list)
    """
    from dataset.custom_dataset import _find_file

    model = model.to(device)
    model.eval()

    # Thư mục ảnh/mask theo split
    img_dir  = os.path.join(data_root, images_dir, split)
    mask_dir = os.path.join(data_root, labels_dir, split)

    # Danh sách file: ưu tiên list_file, fallback scan thư mục
    if list_file and os.path.isfile(list_file):
        with open(list_file, "r") as f:
            stems = [l.strip() for l in f.readlines() if l.strip()]
    else:
        _EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        if os.path.isdir(img_dir):
            stems = sorted([
                os.path.splitext(fn)[0]
                for fn in os.listdir(img_dir)
                if os.path.splitext(fn)[1].lower() in _EXTS
            ])
        else:
            # fallback: flat images dir
            img_dir  = os.path.join(data_root, images_dir)
            mask_dir = os.path.join(data_root, labels_dir)
            stems = sorted([
                os.path.splitext(fn)[0]
                for fn in os.listdir(img_dir)
                if os.path.splitext(fn)[1].lower() in _EXTS
            ])

    is_binary = (num_classes == 2)
    os.makedirs(os.path.dirname(os.path.abspath(save_file)), exist_ok=True)

    results = []
    all_dice      = []
    all_precision = []
    all_recall    = []
    all_iou       = []

    with torch.no_grad():
        for stem in stems:
            # ---- Load ảnh ----
            img_path  = _find_file(img_dir, stem)
            img_orig  = Image.open(img_path).convert("RGB")
            orig_W, orig_H = img_orig.size   # PIL: (W, H)

            img_tensor = img_transform(img_orig).unsqueeze(0).to(device)  # [1,3,H,W]

            # ---- Load mask ----
            mask_stem = stem + mask_suffix
            mask_path = _find_file(mask_dir, mask_stem)
            mask_orig = Image.open(mask_path).convert("L")
            mask_np   = np.array(mask_orig, dtype=np.uint8)
            if is_binary:
                mask_np = (mask_np > 127).astype(np.uint8)
            else:
                mask_np = np.clip(mask_np, 0, num_classes - 1).astype(np.uint8)

            # ---- Inference ----
            pred_logits = model(img_tensor)                          # [1, C, H', W']
            pred = torch.softmax(pred_logits, dim=1).argmax(dim=1)  # [1, H', W']
            pred = pred.squeeze().cpu().numpy().astype(np.uint8)    # [H', W']

            # Resize prediction về kích thước gốc
            pred_resized = np.array(
                Image.fromarray(pred).resize((orig_W, orig_H), resample=Image.NEAREST)
            )

            # ---- Metrics ----
            dice, hd, precision, recall, iou = compute_metrics_per_class(
                pred_resized, mask_np,
                num_classes=num_classes,
                ignore_background=ignore_background,
            )
            results.append((stem, dice, hd, precision, recall, iou))
            all_dice.append(dice)
            all_precision.append(precision)
            all_recall.append(recall)
            all_iou.append(iou)

            hd_str = f"{hd}" if hd is not None else "N/A"
            p_str  = [f"{p*100:.2f}%" for p in precision]
            r_str  = [f"{r*100:.2f}%" for r in recall]
            iou_str= [f"{v*100:.2f}%" for v in iou]
            print(
                f"[{stem}]\n"
                f"  Dice={[f'{d:.4f}' for d in dice]}  HD95={hd_str}\n"
                f"  Precision={p_str}  Recall={r_str}  IoU={iou_str}"
            )

    # ---- Tổng kết ----
    mean_dice      = np.mean(all_dice,      axis=0)
    mean_precision = np.mean(all_precision, axis=0)
    mean_recall    = np.mean(all_recall,    axis=0)
    mean_iou       = np.mean(all_iou,       axis=0)

    overall_dice = float(np.mean(mean_dice))
    overall_prec = float(np.mean(mean_precision))
    overall_rec  = float(np.mean(mean_recall))
    overall_iou  = float(np.mean(mean_iou))

    save_file = os.path.abspath(save_file)
    with open(save_file, "w", encoding="utf-8") as f:
        for stem, d, h, p, r, iou in results:
            f.write(f"{stem} | Dice: {d} | HD95: {h} | Precision: {p} | Recall: {r} | IoU: {iou}\n")
        f.write("\n=== Overall Averages ===\n")
        class_range = range(1, num_classes) if ignore_background else range(num_classes)
        for i, c in enumerate(class_range):
            f.write(
                f"  Class {c}: "
                f"Dice={mean_dice[i]*100:.2f}%  "
                f"Precision={mean_precision[i]*100:.2f}%  "
                f"Recall={mean_recall[i]*100:.2f}%  "
                f"IoU={mean_iou[i]*100:.2f}%\n"
            )
        f.write(
            f"\n  Mean Dice      : {overall_dice*100:.2f}%\n"
            f"  Mean Precision : {overall_prec*100:.2f}%\n"
            f"  Mean Recall    : {overall_rec*100:.2f}%\n"
            f"  Mean IoU (mIoU): {overall_iou*100:.2f}%\n"
        )

    print(f"\n✅ Kết quả lưu tại: {save_file}")
    print(f"   Mean Dice      : {overall_dice*100:.2f}%")
    print(f"   Mean Precision : {overall_prec*100:.2f}%")
    print(f"   Mean Recall    : {overall_rec*100:.2f}%")
    print(f"   Mean IoU (mIoU): {overall_iou*100:.2f}%")

    return results


# ---------------------------------------------------------------------------
# Hàm lưu ảnh kết quả với overlay mask
# ---------------------------------------------------------------------------
def save_overlay_results(
    model,
    data_root: str,
    img_transform,
    save_dir: str = "overlay_results",
    split: str = "test",
    num_classes: int = 2,
    device: str = "cuda",
    alpha: float = 0.4,
    mask_suffix: str = "",
    list_file: str = None,
    images_dir: str = "images",
    labels_dir: str = "labels",
):
    """
    Lưu ảnh overlay giữa prediction và ground truth.

    Màu sắc mặc định (có thể sửa):
        class 1 → đỏ   (prediction) / xanh lá (ground truth)
        class 2 → vàng / xanh dương
        class 3 → tím  / cam
    """
    from dataset.custom_dataset import _find_file

    # Bảng màu per-class [R, G, B]
    CLASS_COLORS_PRED = {
        1: np.array([255,   0,   0], dtype=np.uint8),
        2: np.array([255, 200,   0], dtype=np.uint8),
        3: np.array([150,   0, 255], dtype=np.uint8),
        4: np.array([  0, 200, 255], dtype=np.uint8),
    }
    CLASS_COLORS_GT = {
        1: np.array([  0, 255,   0], dtype=np.uint8),
        2: np.array([  0,   0, 255], dtype=np.uint8),
        3: np.array([255, 128,   0], dtype=np.uint8),
        4: np.array([128, 255, 128], dtype=np.uint8),
    }

    model = model.to(device)
    model.eval()

    img_dir  = os.path.join(data_root, images_dir, split)
    mask_dir = os.path.join(data_root, labels_dir, split)

    if list_file and os.path.isfile(list_file):
        with open(list_file, "r") as f:
            stems = [l.strip() for l in f.readlines() if l.strip()]
    else:
        _EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        if os.path.isdir(img_dir):
            stems = sorted([
                os.path.splitext(fn)[0]
                for fn in os.listdir(img_dir)
                if os.path.splitext(fn)[1].lower() in _EXTS
            ])
        else:
            img_dir  = os.path.join(data_root, images_dir)
            mask_dir = os.path.join(data_root, labels_dir)
            stems = sorted([
                os.path.splitext(fn)[0]
                for fn in os.listdir(img_dir)
                if os.path.splitext(fn)[1].lower() in _EXTS
            ])

    is_binary = (num_classes == 2)
    os.makedirs(save_dir, exist_ok=True)

    def overlay(base_rgb, label_map, color_map):
        out = base_rgb.astype(np.float32)
        for c, color in color_map.items():
            m = (label_map == c)
            if not m.any():
                continue
            m3 = np.stack([m]*3, axis=-1)
            out = out*(1-m3) + ((1-alpha)*out + alpha*color)*m3
        return np.clip(out, 0, 255).astype(np.uint8)

    with torch.no_grad():
        for stem in stems:
            img_path = _find_file(img_dir, stem)
            img_pil  = Image.open(img_path).convert("RGB")
            orig_W, orig_H = img_pil.size
            img_rgb  = np.array(img_pil)

            img_tensor = img_transform(img_pil).unsqueeze(0).to(device)

            # inference
            pred_logits = model(img_tensor)
            pred = torch.softmax(pred_logits, dim=1).argmax(dim=1).squeeze().cpu().numpy()
            pred = np.array(Image.fromarray(pred.astype(np.uint8)).resize((orig_W, orig_H), Image.NEAREST))

            # mask GT
            mask_stem = stem + mask_suffix
            mask_path = _find_file(mask_dir, mask_stem)
            gt = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
            if is_binary:
                gt = (gt > 127).astype(np.uint8)

            # Tạo color map (class 1 nếu binary)
            cmap = {1: CLASS_COLORS_PRED[1]} if is_binary else CLASS_COLORS_PRED
            gmap = {1: CLASS_COLORS_GT[1]}   if is_binary else CLASS_COLORS_GT

            pred_ov = overlay(img_rgb, pred, cmap)
            gt_ov   = overlay(img_rgb, gt,   gmap)

            # Lưu
            Image.fromarray(pred_ov).save(os.path.join(save_dir, f"{stem}_pred.jpg"))
            Image.fromarray(gt_ov).save(os.path.join(save_dir, f"{stem}_gt.jpg"))

    print(f"✅ Overlay ảnh đã lưu tại: {save_dir}/")
