"""
test_custom.py — Script đánh giá DINO-AugSeg trên Custom 2D Dataset
====================================================================
Sử dụng (backbone load từ file .pth local — giống train_custom.py):
    python test_custom.py \
        --data_root   datasets \
        --weight_path checkpoint/dinov3_convnext_small.pth \
        --ckpt        checkpoint/custom/best_iou_model.pth \
        --backbone    convnext_small \
        --num_classes 2 \
        --img_size    512 \
        --split       test \
        --save_file   checkpoint/custom/results_test.txt \
        --save_overlay \
        --overlay_dir checkpoint/custom/overlays

Cấu trúc thư mục dữ liệu:
    <data_root>/
        Images/
            test/     # ảnh test
        labels/
            test/     # mask test

Metrics xuất ra: Precision, Recall, IoU (mIoU), Dice — per-class & mean.
"""

import os
import argparse
import torch
from torchvision import transforms

from train_custom import load_backbone, _BACKBONE_TABLE
from model.Dinov3_WTAUG_UNet import DINO_AugSeg
from utils.eval_custom import test_model_custom, save_overlay_results


def main():
    parser = argparse.ArgumentParser(description="Test DINO-AugSeg on Custom 2D Dataset")

    # ---- Đường dẫn ----
    parser.add_argument("--data_root",   type=str, required=True,
                        help="Thư mục gốc dataset (chứa Images/ và labels/)")
    parser.add_argument("--repo_dir",    type=str, default=".",
                        help="Thư mục gốc repo DINO-AugSeg (chứa hubconf.py)")
    parser.add_argument("--weight_path", type=str, required=True,
                        help="Đường dẫn file .pth pretrained backbone")
    parser.add_argument("--ckpt",        type=str, required=True,
                        help="Checkpoint model đã train (.pth)")
    parser.add_argument("--save_file",   type=str, default="results_test.txt",
                        help="File lưu kết quả metrics")

    # ---- Mô hình ----
    parser.add_argument("--backbone",    type=str, default="convnext_small",
                        choices=list(_BACKBONE_TABLE.keys()),
                        help="Kiến trúc backbone: convnext_tiny/small/base/large | vits16")
    parser.add_argument("--decoder",     type=str, default="cross_guide_wt_unet")

    # ---- Dataset ----
    parser.add_argument("--split",       type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Split để đánh giá")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Số class phân đoạn (bao gồm background)")
    parser.add_argument("--img_size",    type=int, default=512,
                        help="Kích thước ảnh (vuông)")
    parser.add_argument("--mask_suffix", type=str, default="",
                        help="Suffix thêm vào tên file mask (ví dụ '_mask')")
    parser.add_argument("--images_dir",  type=str, default="images",
                        help="Tên thư mục ảnh bên trong data_root")
    parser.add_argument("--labels_dir",  type=str, default="labels",
                        help="Tên thư mục mask bên trong data_root")
    parser.add_argument("--list_file",   type=str, default=None,
                        help="File danh sách stem (None → tự scan thư mục)")

    # ---- Overlay ----
    parser.add_argument("--save_overlay", action="store_true",
                        help="Lưu ảnh overlay prediction vs ground truth")
    parser.add_argument("--overlay_dir",  type=str, default="overlay_results",
                        help="Thư mục lưu overlay ảnh")
    parser.add_argument("--overlay_alpha", type=float, default=0.4,
                        help="Độ trong suốt overlay mask (0=ảnh gốc, 1=màu thuần)")

    # ---- Thiết bị ----
    parser.add_argument("--device",      type=str, default="cuda")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Transform (giống val transform trong train_custom.py) ----
    img_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    # ---- Load backbone (dùng cùng hàm với train_custom.py) ----
    backbone, model_type = load_backbone(
        backbone=args.backbone,
        weight_path=args.weight_path,
        repo_dir=args.repo_dir,
    )

    # ---- Build model ----
    model = DINO_AugSeg(
        encoder=backbone,
        num_classes=args.num_classes,
        model_type=model_type,
        decoder_type=args.decoder,
        use_wt_aug=False,   # Tắt WT-Aug khi testing
        aug_feat=False,
    )

    # ---- Load checkpoint model đã train ----
    print(f"Loading checkpoint: {args.ckpt}")
    state = torch.load(args.ckpt, map_location="cpu")
    # Xử lý checkpoint bọc trong dict
    if isinstance(state, dict):
        for key in ("model", "state_dict"):
            if key in state:
                print(f"  [info] Tìm thấy key '{key}' trong checkpoint.")
                state = state[key]
                break
    model.load_state_dict(state)
    model = model.to(device)
    print("✅ Model loaded.\n")

    # ---- Evaluation: Precision / Recall / IoU / Dice ----
    results = test_model_custom(
        model=model,
        data_root=args.data_root,
        img_transform=img_transform,
        split=args.split,
        num_classes=args.num_classes,
        img_size=args.img_size,
        device=str(device),
        save_file=args.save_file,
        ignore_background=True,
        mask_suffix=args.mask_suffix,
        list_file=args.list_file,
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
    )

    # ---- Overlay (tùy chọn) ----
    if args.save_overlay:
        print(f"\nĐang tạo overlay ảnh → {args.overlay_dir}")
        save_overlay_results(
            model=model,
            data_root=args.data_root,
            img_transform=img_transform,
            save_dir=args.overlay_dir,
            split=args.split,
            num_classes=args.num_classes,
            device=str(device),
            alpha=args.overlay_alpha,
            mask_suffix=args.mask_suffix,
            list_file=args.list_file,
            images_dir=args.images_dir,
            labels_dir=args.labels_dir,
        )


if __name__ == "__main__":
    main()
