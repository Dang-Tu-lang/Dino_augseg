"""
train_custom.py — Script huấn luyện DINO-AugSeg cho Custom 2D Dataset
======================================================================
Sử dụng (backbone tự động tải về nếu chưa có):
    python train_custom.py \
        --data_root   datasets \
        --num_classes 2 \
        --img_size    512 \
        --epochs      100 \
        --batch_size  4 \
        --save_dir    checkpoint/custom

Backbone mặc định: DINOv3 ConvNeXt-Small (tự download ~90 MB lần đầu).
Có thể chọn backbone khác qua --backbone:
    convnext_small  (mặc định, ~90 MB)
    convnext_tiny   (~45 MB)
    convnext_base   (~200 MB)
    vits16          (ViT-Small/16, ~90 MB)

Cấu trúc thư mục dữ liệu cần có:
    <data_root>/
        Images/
            train/    # ảnh training
            val/      # ảnh validation (tùy chọn)
            test/     # ảnh test (tùy chọn)
        labels/
            train/    # mask training
            val/      # mask validation
            test/     # mask test

Kiểm tra thống kê dataset:
    python dataset/custom_dataset.py --data_root datasets --stats
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from dataset.custom_dataset import CustomDataset,EpisodicDataset
from model.Dinov3_WTAUG_UNet import DINO_AugSeg
from utils.eval_custom import compute_segmentation_metrics, _dice_numpy
import cv2
import random
def seed_everything(seed=42):
    # Khóa seed cho Python và các hàm ngẫu nhiên cốt lõi
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Khóa seed cho Numpy (thường dùng trong Data Loader / Augmentation)
    np.random.seed(seed)
    
    # Khóa seed cho PyTorch (CPU & GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # Nếu dùng nhiều GPU
    
    # Ép CuDNN chạy chế độ Tất định (Sẽ làm tốc độ train chậm đi một chút)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Gọi hàm này TRƯỚC KHI khởi tạo DataLoader hay Model
seed_everything(42)
# ---------------------------------------------------------------------------
# Backbone loader — load từ file .pth local
# ---------------------------------------------------------------------------
# Bảng ánh xạ: backbone_name -> (hub_fn_name, model_type cho DINO_AugSeg)
_BACKBONE_TABLE = {
    "convnext_tiny":  ("dinov3_convnext_tiny",  "tiny"),
    "convnext_small": ("dinov3_convnext_small", "small"),
    "convnext_base":  ("dinov3_convnext_base",  "base"),
    "convnext_large": ("dinov3_convnext_large", "large"),
    "vits16":         ("dinov3_vits16",         "small"),
}


def load_backbone(backbone: str, weight_path: str, repo_dir: str):
    """
    Khởi tạo kiến trúc backbone DINOv3 (không load pretrained từ internet),
    sau đó nạp state dict từ file .pth local.

    Args:
        backbone     : tên backbone, xem _BACKBONE_TABLE.
        weight_path  : đường dẫn đến file .pth chứa state dict backbone.
        repo_dir     : thư mục gốc repo DINO-AugSeg (chứa hubconf.py).

    Returns:
        (backbone_module, model_type_str)
    """
    if backbone not in _BACKBONE_TABLE:
        raise ValueError(
            f"Backbone '{backbone}' không hợp lệ. Chọn một trong: "
            + ", ".join(_BACKBONE_TABLE.keys())
        )

    if not os.path.isfile(weight_path):
        raise FileNotFoundError(
            f"Không tìm thấy file weight: '{weight_path}'\n"
            f"Hãy chỉ đúng đường dẫn qua --weight_path"
        )

    hub_fn, model_type = _BACKBONE_TABLE[backbone]

    print(f"\n{'='*60}")
    print(f"  Backbone    : {backbone}  ({hub_fn})")
    print(f"  Weight file : {weight_path}")
    print(f"{'='*60}\n")

    # Khởi tạo kiến trúc (không download weight)
    backbone_model = torch.hub.load(
        repo_or_dir=repo_dir,
        model=hub_fn,
        source="local",
        pretrained=False,
    )

    # Nạp state dict từ file .pth
    state_dict = torch.load(weight_path, map_location="cpu")
    # Một số checkpoint có thể bọc trong dict với key 'model' hoặc 'state_dict'
    if isinstance(state_dict, dict):
        for key in ("model", "state_dict", "backbone", "encoder"):
            if key in state_dict:
                print(f"  [info] Tìm thấy key '{key}' trong checkpoint, dùng sub-dict này.")
                state_dict = state_dict[key]
                break

    missing, unexpected = backbone_model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [warn] Missing keys  ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"  [warn] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")

    backbone_model.eval()
    n_params = sum(p.numel() for p in backbone_model.parameters()) / 1e6
    print(f"✅ Đã load backbone: {hub_fn}  ({n_params:.1f} M params)")
    return backbone_model, model_type


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
class DiceLoss(nn.Module):
    """Dice Loss cho binary segmentation."""
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred: [B,1,H,W] logits | target: [B,H,W] long
        pred = torch.sigmoid(pred.squeeze(1))  # [B,H,W]
        target = target.float()
        inter = (pred * target).sum()
        dice  = (2.0 * inter + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1.0 - dice


class MultiClassDiceLoss(nn.Module):
    """Dice Loss cho multi-class segmentation (Dynamic)."""
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        # preds: [B,C,H,W] logits | targets: [B,H,W] long
        preds = torch.softmax(preds, dim=1)
        
        valid_mask = (targets != 255)
        targets_safe = targets.clone()
        targets_safe[~valid_mask] = 0
        
        dynamic_num_classes = preds.shape[1]
        targets_oh = F.one_hot(targets_safe, dynamic_num_classes).permute(0,3,1,2).float()
        
        valid_mask = valid_mask.unsqueeze(1).float()
        preds = preds * valid_mask
        targets_oh = targets_oh * valid_mask
        
        dims = (0, 2, 3)
        inter = torch.sum(preds * targets_oh, dims)
        card  = torch.sum(preds + targets_oh, dims)
        dice  = (2.0 * inter + self.smooth) / (card + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """CrossEntropy + Dice Loss (Dynamic)."""
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(ignore_index=255)
        self.dice = MultiClassDiceLoss()

    def forward(self, preds, targets):
        ce_loss   = self.ce(preds, targets)
        dice_loss = self.dice(preds, targets)
        return self.alpha * ce_loss + (1.0 - self.alpha) * dice_loss

def lovasz_grad(gt_sorted):
    """Tính toán gradient của Lovasz extension dựa trên các lỗi đã được sắp xếp."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # Xử lý trường hợp có 1 pixel
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

class LovaszSoftmaxLoss(nn.Module):
    """Lovasz-Softmax Loss tối ưu trực tiếp cho chỉ số mIoU."""
    def __init__(self, ignore_index: int = 255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, preds, targets):
        # preds: [B, C, H, W] logits | targets: [B, H, W] long
        preds = F.softmax(preds, dim=1)
        
        # Dàn phẳng (flatten)
        preds = preds.permute(0, 2, 3, 1).contiguous().view(-1, preds.size(1))
        targets = targets.view(-1)
        
        # Lọc bỏ rác ranh giới
        valid_mask = targets != self.ignore_index
        if valid_mask.sum() == 0:
            return preds.sum() * 0.0

        preds = preds[valid_mask]
        targets = targets[valid_mask]

        loss = 0.0
        # DUYỆT QUA TẤT CẢ CLASS (không chỉ unique(targets))
        # → Nếu model dự đoán ra class C mà targets không có C,
        #   errors vẫn lớn → Loss vẫn phạt → giảm False Positives.
        for c in range(preds.size(1)):
            fg = (targets == c).float() # Mảng nhị phân cho class hiện tại
            errors = (fg - preds[:, c]).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            fg_sorted = fg[perm]
            loss += torch.dot(errors_sorted, lovasz_grad(fg_sorted))
            
        return loss / preds.size(1)

# ---------------------------------------------------------------------------
# OHEM Cross Entropy Loss
# ---------------------------------------------------------------------------
class OHEMCrossEntropyLoss(nn.Module):
    """OHEM CE ép mô hình học các pixel khó đoán nhất."""
    def __init__(self, ignore_index: int = 255, keep_ratio: float = 0.8, weight=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.keep_ratio = keep_ratio
        # Dùng reduction='none' để lấy loss của từng pixel
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='none')

    def forward(self, preds, targets):
        # Tính loss thô cho toàn bộ pixel
        loss = self.criterion(preds, targets) # [B, H, W]
        
        # Lọc bỏ vùng ignore
        mask = targets != self.ignore_index
        loss = loss[mask]

        if loss.numel() == 0:
            return loss.sum() * 0.0

        # Lấy Top K% các pixel có loss cao nhất (khó đoán nhất)
        num_keep = int(self.keep_ratio * loss.numel())
        if num_keep > 0:
            loss, _ = torch.topk(loss, num_keep)

        return loss.mean()
class FocalLoss(nn.Module):
    """
    Focal Loss: Tập trung vào các pixel khó đoán mà KHÔNG vứt bỏ dữ liệu như OHEM.
    Tuyệt vời cho Few-Shot Imbalanced.
    """
    def __init__(self, alpha=1.0, gamma=2.0, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs: [B, C, H, W] logits | targets: [B, H, W]
        ce_loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss) # Tính xác suất của class đúng
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        
        # Lọc bỏ vùng ignore (viền/background vô ích)
        mask = targets != self.ignore_index
        if mask.sum() == 0:
            return focal_loss.sum() * 0.0
            
        return focal_loss[mask].mean()
# ---------------------------------------------------------------------------
# Loss Tích Hợp (Kết hợp OHEM và Lovasz)
# ---------------------------------------------------------------------------
class CombinedHardLoss(nn.Module):
    """
    Kết hợp OHEM Cross-Entropy và Lovasz-Softmax.
    Đây là setup "hạng nặng" để trị mất cân bằng class và tăng mIoU.
    """
    def __init__(self, ignore_index: int = 255, alpha: float = 0.5, class_weights=None, keep_ratio: float = 0.8):
        super().__init__()
        self.alpha = alpha
        
        # Khởi tạo 2 loss thành phần
        self.ohem_ce = OHEMCrossEntropyLoss(ignore_index=ignore_index, keep_ratio=keep_ratio, weight=class_weights)
        self.lovasz = LovaszSoftmaxLoss(ignore_index=ignore_index)
        self.focus = FocalLoss(ignore_index=ignore_index, gamma=2.0)

    def forward(self, preds, targets):
        # preds: [B, C, H, W] | targets: [B, H, W]
        # ohem_loss = self.ohem_ce(preds, targets)
        lovasz_loss = self.lovasz(preds, targets)
        focus_loss = self.focus(preds, targets)
        # alpha điều chỉnh tỷ trọng giữa phân loại đúng (OHEM) và đường viền đẹp (Lovasz)
        # print(f"Total: {(self.alpha * ohem_loss + (1.0 - self.alpha) * lovasz_loss).item():.4f} | OHEM: {ohem_loss.item():.4f} | Lovasz: {lovasz_loss.item():.4f}")
        return self.alpha * focus_loss + (1.0 - self.alpha) * lovasz_loss
# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Training loop (K-Shot Episodic + Gradient Accumulation cho GPU mạnh)
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None,
                    grad_accum_steps=1, split_size=3):
    """
    Episodic training với Gradient Accumulation.
    - grad_accum_steps: Số episode cộng dồn gradient trước khi step.
                        Hiệu quả tương đương batch = grad_accum_steps.
    - split_size:       Số ảnh support xử lý cùng lúc khi trích prototype.
                        GPU mạnh (Blackwell) có thể đẩy lên 8-15.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    optimizer.zero_grad()  # Zero grad 1 lần ở đầu

    pbar = tqdm(enumerate(loader), total=len(loader), desc="Training", leave=False)
    for step_idx, (s_imgs, s_masks, q_imgs, q_masks) in pbar:
        # Xóa chiều batch (vì DataLoader batch_size=1)
        s_imgs, s_masks = s_imgs.squeeze(0), s_masks.squeeze(0)
        q_imgs, q_masks = q_imgs.squeeze(0), q_masks.squeeze(0)

        # ==========================================
        # 1. TRÍCH XUẤT ĐẶC TRƯNG TỪ SUPPORT SET (No Grad)
        # ==========================================
        s_feat_list = []
        s_mask_list = []

        model.eval()  # Tắt augmentation cho support
        with torch.no_grad():
            for i in range(0, s_imgs.size(0), split_size):
                mini_s = s_imgs[i : i+split_size].to(device)
                mini_m = s_masks[i : i+split_size].to(device)
                mini_feat, mini_mask_r = model.extract_support_features(mini_s, mini_m)
                s_feat_list.append(mini_feat)
                s_mask_list.append(mini_mask_r)
        model.train() # Bật lại augmentation cho query

        support_features = [torch.cat([s[i] for s in s_feat_list], dim=0) for i in range(4)]
        support_masks_r = torch.cat(s_mask_list, dim=0)

        # ==========================================
        # 2. HỌC TỪ QUERY SET (With Grad)
        #    Loss chia cho grad_accum_steps để giữ scale gradient ổn định
        # ==========================================
        q_imgs = q_imgs.to(device)
        q_masks = q_masks.to(device)

        if scaler is not None:
            from torch.cuda.amp import autocast
            with autocast():
                preds = model(q_imgs, support_features=support_features, support_masks=support_masks_r, targets=q_masks)
                loss  = criterion(preds, q_masks) / grad_accum_steps
            scaler.scale(loss).backward()
        else:
            preds = model(q_imgs, support_features=support_features, support_masks=support_masks_r, targets=q_masks)
            loss  = criterion(preds, q_masks) / grad_accum_steps
            loss.backward()

        current_loss_val = loss.item() * grad_accum_steps
        pbar.set_postfix(loss=f"{current_loss_val:.4f}")
        
        total_loss += current_loss_val * q_imgs.size(0)
        total_samples += q_imgs.size(0)

        # ==========================================
        # 3. STEP OPTIMIZER sau mỗi grad_accum_steps episodes
        # ==========================================
        if (step_idx + 1) % grad_accum_steps == 0 or (step_idx + 1) == len(loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

    return total_loss / max(total_samples, 1)

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    pbar = tqdm(loader, desc="Validation", leave=False)
    for s_imgs, s_masks, q_imgs, q_masks in pbar:
        s_imgs, s_masks = s_imgs.squeeze(0).to(device), s_masks.squeeze(0).to(device)
        q_imgs, q_masks = q_imgs.squeeze(0).to(device), q_masks.squeeze(0).to(device)
        
        # Vì là Validation, có thể cho tất cả support vào chung nếu đủ RAM, hoặc chia split như train
        support_features, support_masks_r = model.extract_support_features(s_imgs, s_masks)
        
        preds = model(q_imgs, support_features=support_features, support_masks=support_masks_r, targets=q_masks)
        loss  = criterion(preds, q_masks)
        total_loss += loss.item() * q_imgs.size(0)
        total_samples += q_imgs.size(0)
        
    return total_loss / max(total_samples, 1)

@torch.no_grad()
def validate_metrics(model, loader, device, num_classes: int = 7,
                     ignore_background: bool = True, dilation_ratio: float = 0.02):
    model.eval()
    is_binary = (num_classes == 2)
    class_range = list(range(1, num_classes)) if ignore_background else list(range(num_classes))
    n_cls = len(class_range)

    tp_arr   = np.zeros(n_cls, dtype=np.float64)
    fp_arr   = np.zeros(n_cls, dtype=np.float64)
    fn_arr   = np.zeros(n_cls, dtype=np.float64)
    dice_num = np.zeros(n_cls, dtype=np.float64)
    dice_den = np.zeros(n_cls, dtype=np.float64)

    # 1. THAY ĐỔI LOADER: Nhận 4 biến từ EpisodicDataset
    pbar = tqdm(loader, desc="Eval Metrics", leave=False)
    for s_imgs, s_masks, q_imgs, q_masks in pbar:
        # Squeeze bỏ chiều batch=1 của Episodic DataLoader
        s_imgs  = s_imgs.squeeze(0).to(device)
        s_masks = s_masks.squeeze(0).to(device)
        q_imgs  = q_imgs.squeeze(0).to(device)
        q_masks = q_masks.squeeze(0).to(device) 

        # =======================================================
        # 2. TÍNH GLOBAL PROTOTYPES TỪ 15 ẢNH SUPPORT (Chống OOM)
        # =======================================================
        s_feat_list = []
        s_mask_list = []
        split_size = 3 # Chunking an toàn VRAM
        
        for i in range(0, s_imgs.size(0), split_size):
            mini_s = s_imgs[i : i+split_size]
            mini_m = s_masks[i : i+split_size]
            
            mini_f, mini_m_r = model.extract_support_features(mini_s, mini_m)
            s_feat_list.append(mini_f)
            s_mask_list.append(mini_m_r)

        support_features = [torch.cat([s[i] for s in s_feat_list], dim=0) for i in range(4)]
        support_masks_r = torch.cat(s_mask_list, dim=0)

        # =======================================================
        # 3. DỰ ĐOÁN VÀ ĐÁNH GIÁ TRÊN ẢNH QUERY
        # =======================================================
        preds = model(q_imgs, support_features=support_features, support_masks=support_masks_r)

        if is_binary:
            pred_cls = torch.softmax(preds, dim=1).argmax(dim=1)
        else:
            pred_cls = torch.softmax(preds, dim=1).argmax(dim=1)

        pred_np  = pred_cls.cpu().numpy().astype(np.uint8)
        mask_np  = q_masks.cpu().numpy().astype(np.uint8) # SO SÁNH VỚI Q_MASKS

        for b in range(pred_np.shape[0]):
            p = pred_np[b]
            t = mask_np[b]
            
            valid_mask = (t != 255)

            for i, c in enumerate(class_range):
                pred_c   = ((p == c) & valid_mask).astype(np.uint8)
                target_c = ((t == c) & valid_mask).astype(np.uint8)
                
                # --- Metrics gốc (mIoU, P, R) ---
                tp = (pred_c & target_c).sum()
                fp = (pred_c & ~target_c).sum()
                fn = (~pred_c & target_c).sum()
                tp_arr[i]  += tp
                fp_arr[i]  += fp
                fn_arr[i]  += fn
                dice_num[i] += 2 * tp
                dice_den[i] += pred_c.sum() + target_c.sum()

    smooth = 1e-8
    precision_list = (tp_arr / (tp_arr + fp_arr + smooth)).tolist()
    recall_list    = (tp_arr / (tp_arr + fn_arr + smooth)).tolist()
    iou_list       = (tp_arr / (tp_arr + fp_arr + fn_arr + smooth)).tolist()
    dice_list      = ((dice_num + smooth) / (dice_den + smooth)).tolist()

    return {
        "precision" : precision_list,
        "recall"    : recall_list,
        "iou"       : iou_list,
        "dice"      : dice_list,
        "mean_precision" : float(np.mean(precision_list)),
        "mean_recall"    : float(np.mean(recall_list)),
        "mean_iou"       : float(np.mean(iou_list)),
        "mean_dice"      : float(np.mean(dice_list)),
    }







# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train DINO-AugSeg on Custom 2D Dataset")

    # Đường dẫn
    parser.add_argument("--data_root",   type=str, required=True,
                        help="Thư mục gốc của dataset (chứa Images/ và labels/)")
    parser.add_argument("--repo_dir",    type=str, default=".",
                        help="Thư mục gốc repo DINO-AugSeg (chứa hubconf.py)")
    parser.add_argument("--weight_path", type=str, required=True,
                        help="Đường dẫn đến file .pth chứa pretrained weights của backbone")
    parser.add_argument("--save_dir",    type=str, default="checkpoint/custom",
                        help="Thư mục lưu checkpoint")

    # Mô hình
    parser.add_argument("--backbone",    type=str, default="convnext_small",
                        choices=list(_BACKBONE_TABLE.keys()),
                        help="Kiến trúc backbone: convnext_tiny/small/base/large | vits16")
    parser.add_argument("--decoder",     type=str, default="cross_guide_wt_unet")
    parser.add_argument("--use_wt_aug",  action="store_true", default=True,
                        help="Dùng Wavelet augmentation trong training")
    parser.add_argument("--no_wt_aug",   dest="use_wt_aug", action="store_false")

    # Dataset
    parser.add_argument("--num_classes", type=int, default=7,
                        help="Số lớp phân đoạn (bao gồm background, ví dụ 2 cho binary)")
    parser.add_argument("--n_way",       type=int, default=3,
                        help="Số foreground class sample mỗi episode cho N-way few-shot")
    parser.add_argument("--train_classes", type=int, nargs="+", default=None,
                        help="Danh sách ID các class dùng để Train (ví dụ: 1 2 3 4 5)")
    parser.add_argument("--val_classes", type=int, nargs="+", default=None,
                        help="Danh sách ID các class dùng để Validation (ví dụ: 6 7)")
    parser.add_argument("--img_size",    type=int, default=512,
                        help="Kích thước ảnh (vuông)")
    parser.add_argument("--train_num",   default="all",
                        help="Số mẫu tối đa cho training ('all' = dùng hết)")
    parser.add_argument("--mask_suffix", type=str, default="_labelTrainIds",
                        help="Suffix thêm vào tên file mask (ví dụ '_mask')")
    parser.add_argument("--images_dir",  type=str, default="images",
                        help="Tên thư mục ảnh bên trong data_root (mặc định 'Images')")
    parser.add_argument("--labels_dir",  type=str, default="labels",
                        help="Tên thư mục mask bên trong data_root (mặc định 'labels')")

    # Hyperparameters
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--lr",          type=float, default=5e-5)
    parser.add_argument("--alpha",       type=float, default=0.5,
                        help="Trọng số CE trong CombinedLoss (0=Dice only, 1=CE only)")
    parser.add_argument("--amp",         action="store_true",
                        help="Dùng Automatic Mixed Precision (tiết kiệm VRAM)")
    parser.add_argument("--num_workers", type=int, default=4)

    # GPU Utilization — Tận dụng sức mạnh GPU hạng nặng (Blackwell, H100, ...)
    parser.add_argument("--grad_accum",  type=int, default=1,
                        help="Gradient Accumulation: cộng dồn N episodes trước khi step. "
                             "Hiệu quả = batch_size * grad_accum. VD: --grad_accum 8")
    parser.add_argument("--num_support", type=int, default=5,
                        help="Số ảnh support mỗi episode (mặc định 5, có thể 10-15)")
    parser.add_argument("--num_query",   type=int, default=4,
                        help="Số ảnh query mỗi episode (mặc định 4, GPU mạnh đẩy 8-16)")
    parser.add_argument("--split_size",  type=int, default=3,
                        help="Chunk size khi trích prototype support (mặc định 3, GPU mạnh đẩy 8-15)")
    parser.add_argument("--episodes_per_epoch", type=int, default=400,
                        help="Số episodes mỗi epoch training")

    args = parser.parse_args()

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # ---- Transform ----
    img_transform_train = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    img_transform_val = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    # ---- Dataset ----
    train_num = "all" if args.train_num == "all" else int(args.train_num)
    base_train_dataset = CustomDataset(
        data_root=args.data_root, split="train",
        num_classes=args.num_classes, transform_img=img_transform_train,
        img_size=args.img_size, train_num=train_num,
        mask_suffix=args.mask_suffix,
        images_dir=args.images_dir, labels_dir=args.labels_dir,
    )
    train_dataset = EpisodicDataset(base_train_dataset, num_support=args.num_support,
                                    num_query=args.num_query, episodes_per_epoch=args.episodes_per_epoch, n_way=args.n_way, valid_classes=args.train_classes)
    train_loader = DataLoader(
        train_dataset, batch_size=1,
        shuffle=True, num_workers=args.num_workers, pin_memory=True,
    )
    print(f"\n🚀 GPU Utilization Config:")
    print(f"   Support/episode : {args.num_support}")
    print(f"   Query/episode   : {args.num_query}")
    print(f"   Grad Accumulation: {args.grad_accum} episodes")
    print(f"   Effective batch : {args.num_query} × {args.grad_accum} = {args.num_query * args.grad_accum} query images/step")
    print(f"   Split size      : {args.split_size}")
    print(f"   Episodes/epoch  : {args.episodes_per_epoch}\n")

    # Val loader (tùy chọn)
    val_loader = None
    val_img_dir = os.path.join(args.data_root, args.images_dir, "val")
    if os.path.isdir(val_img_dir) and any(
        os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        for f in os.listdir(val_img_dir)
    ):
        base_val_dataset = CustomDataset(
            data_root=args.data_root, split="val",
            num_classes=args.num_classes, transform_img=img_transform_val,
            img_size=args.img_size, mask_suffix=args.mask_suffix,
            images_dir=args.images_dir, labels_dir=args.labels_dir,
        )
        if args.val_classes is not None:
            print(f"✅ Tạo Validation Loader dùng class {args.val_classes} trên khối dữ liệu Val.")
            val_dataset = EpisodicDataset(base_val_dataset, num_support=args.num_support,
                                          num_query=args.num_query, episodes_per_epoch=100, deterministic=True, n_way=args.n_way, valid_classes=args.val_classes)
        else:
            val_dataset = EpisodicDataset(base_val_dataset, num_support=args.num_support,
                                          num_query=args.num_query, episodes_per_epoch=100, deterministic=True, n_way=args.n_way)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # ---- Backbone: load từ file .pth ----
    backbone, model_type = load_backbone(
        backbone=args.backbone,
        weight_path=args.weight_path,
        repo_dir=args.repo_dir,
    )

    # ---- Model ----
    model = DINO_AugSeg(
        encoder=backbone,
        num_classes=args.n_way + 1,
        model_type=model_type,          # lấy từ backbone table, không cần truyền tay
        # decoder_type=args.decoder,
        use_wt_aug=args.use_wt_aug,
        # aug_feat=False,
    ).to(device)
    # ---- Optimizer & Loss ----
    # Chỉ tối ưu decoder (encoder đã được freeze trong DINO_AugSeg.__init__)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")

    optimizer = optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    warmup_epochs = 15
    scheduler_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs-warmup_epochs, eta_min=1e-6)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs])
    criterion = CombinedLoss(alpha=args.alpha)

    scaler = None
    if args.amp and torch.cuda.is_available():
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
        print("AMP enabled.")

    # ---- Training Loop ----
    log_path = os.path.join(args.save_dir, "train_log.txt")
    best_val_loss = float("inf")
    best_val_iou  = -1.0
    patience = 25 
    epochs_no_improve = 0
    # Tần suất tính metrics nặng (P/R/IoU/Dice) trên val set
    METRICS_EVERY = max(1, args.epochs // 100)  # tối đa ~20 lần trong quá trình train

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        
        if epoch < args.epochs * 0.5:
            # 50% thời gian đầu: Giữ nguyên mức nhiễu tối đa để chống Overfitting
            current_prob = 0.7
        else:
            # 50% thời gian sau: Giảm tuyến tính từ 0.7 về 0 để gọt đường viền sắc nét
            current_prob = 0.7 * (1.0 - (epoch - args.epochs * 0.5) / (args.epochs * 0.5))
        model.update_random_choice(current_prob)
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler,
                                     grad_accum_steps=args.grad_accum, split_size=args.split_size)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        log_line = f"Epoch {epoch}/{args.epochs} [{epoch_duration:.1f}s] | LR: {current_lr:.6e} | Train Loss: {train_loss:.4f}"

        if val_loader is not None:
            val_loss = validate(model, val_loader, criterion, device)
            log_line += f" | Val Loss: {val_loss:.4f}"

            # Tính P / R / IoU / Dice định kỳ
            # Sau N-way remap, chỉ có classes 0..n_way → num_classes = n_way + 1
            if epoch % METRICS_EVERY == 0 or epoch == args.epochs:
                m = validate_metrics(
                    model, val_loader, device,
                    num_classes=args.n_way + 1,
                    ignore_background=True,
                )
                mP  = m["mean_precision"] * 100
                mR  = m["mean_recall"]    * 100
                mIoU= m["mean_iou"]       * 100
                mD  = m["mean_dice"]      * 100
                metrics_str = (
                    f" | mP={mP:.2f}% mR={mR:.2f}%"
                    f" mIoU={mIoU:.2f}% mDice={mD:.2f}%"
                )
                log_line += metrics_str

                # Lưu best theo mIoU
                if m["mean_iou"] > best_val_iou:
                    best_val_iou = m["mean_iou"]
                    best_iou_ckpt = os.path.join(args.save_dir, "best_iou_model.pth")
                    torch.save(model.state_dict(), best_iou_ckpt)
                    log_line += " ← best IoU"
                    epochs_no_improve = 0 
                else:
                    epochs_no_improve +=1
            # Lưu checkpoint tốt nhất theo loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt = os.path.join(args.save_dir, "best_model.pth")
                torch.save(model.state_dict(), best_ckpt)
                log_line += " ← best loss"

        print(log_line)
        with open(log_path, "a") as f:
            f.write(log_line + "\n")
        if epochs_no_improve >= patience:
            stop_msg = f"🛑 Early stopping kích hoạt tại epoch {epoch}. mIoU không cải thiện trong {patience} epoch."
            print(stop_msg)
            with open(log_path, "a") as f:
                f.write(stop_msg + "\n")
            break
    # ---- Lưu model epoch cuối ----
    last_ckpt = os.path.join(args.save_dir, "last_model.pth")
    torch.save(model.state_dict(), last_ckpt)
    print(f"\n✅ Saved last model → {last_ckpt}")
    if val_loader is not None:
        print(f"✅ Best loss model → {os.path.join(args.save_dir, 'best_model.pth')}")
        print(f"✅ Best IoU  model → {os.path.join(args.save_dir, 'best_iou_model.pth')}")

    # ---- Final evaluation trên val set ----
    test_loader = None
    test_img_dir = os.path.join(args.data_root, args.images_dir, "test")
    if os.path.isdir(test_img_dir) and any(
        os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        for f in os.listdir(test_img_dir)
    ):
        base_test_dataset = CustomDataset(
            data_root=args.data_root, split="test",
            num_classes=args.num_classes, transform_img=img_transform_val,
            img_size=args.img_size, mask_suffix=args.mask_suffix,
            images_dir=args.images_dir, labels_dir=args.labels_dir,
        )
        test_dataset = EpisodicDataset(base_test_dataset, num_support=args.num_support,
                                       num_query=args.num_query, episodes_per_epoch=100, deterministic=True)
        test_loader = DataLoader(
            test_dataset, batch_size=1,
            shuffle=False, num_workers=args.num_workers, pin_memory=True,
        )
    if test_loader is not None:
        print("\n" + "="*60)
        print("  Final Validation Metrics (full dataset)")
        print("="*60)
        m = validate_metrics(
            model, test_loader, device,
            num_classes=args.n_way + 1,
            ignore_background=True,
        )
        class_range = range(1, args.n_way + 1)
        for i, c in enumerate(class_range):
            print(
                f"  Class {c}: "
                f"Precision={m['precision'][i]*100:.2f}%  "
                f"Recall={m['recall'][i]*100:.2f}%  "
                f"IoU={m['iou'][i]*100:.2f}%  "
                f"Dice={m['dice'][i]*100:.2f}%"
            )
        print(
            f"  ── Mean ──"
            f"  Precision={m['mean_precision']*100:.2f}%"
            f"  Recall={m['mean_recall']*100:.2f}%"
            f"  mIoU={m['mean_iou']*100:.2f}%"
            f"  mDice={m['mean_dice']*100:.2f}%"
        )
        # Ghi vào log
        with open(log_path, "a") as f:
            f.write("\n=== Final Val Metrics ===\n")
            for i, c in enumerate(class_range):
                f.write(
                    f"  Class {c}: P={m['precision'][i]*100:.2f}%  "
                    f"R={m['recall'][i]*100:.2f}%  "
                    f"IoU={m['iou'][i]*100:.2f}%  "
                    f"Dice={m['dice'][i]*100:.2f}%\n"
                )
            f.write(
                f"  Mean: P={m['mean_precision']*100:.2f}%  "
                f"R={m['mean_recall']*100:.2f}%  "
                f"mIoU={m['mean_iou']*100:.2f}%  "
                f"mDice={m['mean_dice']*100:.2f}%\n"
            )


if __name__ == "__main__":
    main()


