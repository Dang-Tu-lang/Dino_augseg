import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.custom_dataset import CustomDataset, EpisodicDataset
from model.Dinov3_WTAUG_UNet import DINO_AugSeg
from train_custom import load_backbone, validate_metrics, _BACKBONE_TABLE

def main():
    parser = argparse.ArgumentParser(description="Test DINO-AugSeg on Custom 2D Dataset")

    parser.add_argument("--data_root",   type=str, required=True, help="Thư mục gốc của dataset (chứa Images/ và labels/)")
    parser.add_argument("--repo_dir",    type=str, default=".", help="Thư mục gốc repo DINO-AugSeg")
    parser.add_argument("--backbone_weight", type=str, required=True, help="Đường dẫn đến file .pth chứa pretrained weights của backbone")
    parser.add_argument("--ckpt_path",   type=str, required=True, help="Đường dẫn đến checkpoint đã train xong (VD: checkpoint/custom/best_iou_model.pth)")
    
    parser.add_argument("--backbone",    type=str, default="convnext_small", choices=list(_BACKBONE_TABLE.keys()))
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--img_size",    type=int, default=512)
    parser.add_argument("--mask_suffix", type=str, default="_labelTrainIds")
    parser.add_argument("--images_dir",  type=str, default="images")
    parser.add_argument("--labels_dir",  type=str, default="labels")
    parser.add_argument("--use_wt_aug",  action="store_true", default=True)
    
    parser.add_argument("--num_support", type=int, default=5, help="Số ảnh support mỗi episode")
    parser.add_argument("--num_query",   type=int, default=4, help="Số ảnh query mỗi episode")
    parser.add_argument("--episodes",    type=int, default=100, help="Số episodes để test")
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Transform & Dataset
    img_transform_val = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    test_img_dir = os.path.join(args.data_root, args.images_dir, "test")
    if not os.path.isdir(test_img_dir):
        print(f"❌ Không tìm thấy thư mục test tại: {test_img_dir}")
        return

    print("Đang khởi tạo DataLoader...")
    base_test_dataset = CustomDataset(
        data_root=args.data_root, split="test",
        num_classes=args.num_classes, transform_img=img_transform_val,
        img_size=args.img_size, mask_suffix=args.mask_suffix,
        images_dir=args.images_dir, labels_dir=args.labels_dir,
    )
    test_dataset = EpisodicDataset(
        base_test_dataset, num_support=args.num_support,
        num_query=args.num_query, episodes_per_epoch=args.episodes, deterministic=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )

    # 2. Load Backbone & Model
    backbone, model_type = load_backbone(backbone=args.backbone, weight_path=args.backbone_weight, repo_dir=args.repo_dir)
    
    model = DINO_AugSeg(
        encoder=backbone,
        num_classes=args.num_classes,
        model_type=model_type,
        use_wt_aug=args.use_wt_aug,
    ).to(device)

    # 3. Load Checkpoint
    if not os.path.isfile(args.ckpt_path):
        print(f"❌ Không tìm thấy checkpoint tại: {args.ckpt_path}")
        return
        
    print(f"Đang nạp trọng số đã huấn luyện từ: {args.ckpt_path}")
    state_dict = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # Ép prob của random mask về 0 khi test (để đường viền ổn định nhất)
    model.update_random_choice(0.0)
    model.eval()

    # 4. Evaluate
    print("\n" + "="*60)
    print(f"  BẮT ĐẦU TEST TRÊN {args.episodes} EPISODES")
    print("="*60)
    
    m = validate_metrics(
        model, test_loader, device,
        num_classes=args.num_classes,
        ignore_background=True,
    )
    
    class_range = range(1, args.num_classes)
    for i, c in enumerate(class_range):
        print(
            f"  Class {c}: "
            f"P={m['precision'][i]*100:.2f}%  "
            f"R={m['recall'][i]*100:.2f}%  "
            f"IoU={m['iou'][i]*100:.2f}%  "
            f"Dice={m['dice'][i]*100:.2f}%"
        )
    print(
        f"\n  ── TỔNG KẾT (MEAN) ──\n"
        f"  mPrecision = {m['mean_precision']*100:.2f}%\n"
        f"  mRecall    = {m['mean_recall']*100:.2f}%\n"
        f"  mIoU       = {m['mean_iou']*100:.2f}%\n"
        f"  mDice      = {m['mean_dice']*100:.2f}%\n"
        f"============================================================"
    )

if __name__ == "__main__":
    main()
