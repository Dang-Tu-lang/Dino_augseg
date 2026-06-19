import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from dataset.custom_dataset import CustomDataset, EpisodicDataset
from model.Dinov3_WTAUG_UNet import DINO_AugSeg
from train_custom import load_backbone, validate_metrics, _BACKBONE_TABLE

COLOR_PALETTE = [
    [0, 0, 0],       # 0: Background
    [255, 0, 0],     # 1: Red
    [0, 255, 0],     # 2: Green
    [0, 150, 255],   # 3: Blue
    [255, 255, 0],   # 4: Yellow
    [255, 0, 255],   # 5: Magenta
    [0, 255, 255],   # 6: Cyan
    [128, 0, 0],     # 7: Maroon
    [0, 128, 0],     # 8: Dark Green
    [0, 0, 128],     # 9: Navy
    [128, 128, 0],   # 10: Olive
    [128, 0, 128],   # 11: Purple
    [0, 128, 128],   # 12: Teal
    [255, 165, 0],   # 13: Orange
    [255, 192, 203], # 14: Pink
    [165, 42, 42],   # 15: Brown
    [210, 105, 30],  # 16: Chocolate
    [255, 99, 71],   # 17: Tomato
    [255, 20, 147],  # 18: Deep Pink
    [199, 21, 133],  # 19: Medium Violet Red
    [139, 0, 139],   # 20: Dark Magenta
    [75, 0, 130],    # 21: Indigo
    [72, 61, 139],   # 22: Dark Slate Blue
    [0, 0, 205],     # 23: Medium Blue
    [30, 144, 255],  # 24: Dodger Blue
    [0, 191, 255],   # 25: Deep Sky Blue
    [32, 178, 170],  # 26: Light Sea Green
    [60, 179, 113],  # 27: Medium Sea Green
    [34, 139, 34],   # 28: Forest Green
    [154, 205, 50],  # 29: Yellow Green
    [85, 107, 47],   # 30: Dark Olive Green
    [189, 183, 107], # 31: Dark Khaki
    [240, 230, 140], # 32: Khaki
    [184, 134, 11],  # 33: Dark Goldenrod
    [218, 165, 32],  # 34: Goldenrod
    [244, 164, 96],  # 35: Sandy Brown
    [250, 128, 114], # 36: Salmon
    [233, 150, 122], # 37: Dark Salmon
    [240, 128, 128], # 38: Light Coral
    [205, 92, 92],   # 39: Indian Red
    [178, 34, 34]    # 40: Firebrick
]

def colorize_mask(mask_tensor, num_classes=7):
    mask_np = mask_tensor.cpu().numpy()
    h, w = mask_np.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(1, min(num_classes, len(COLOR_PALETTE))): 
        color_mask[mask_np == c] = COLOR_PALETTE[c]
    return color_mask

def unnormalize(tensor):
    """Đưa ảnh từ dạng chuẩn hóa (-1 tới 1) về lại dạng RGB (0 tới 1) để lưu ảnh"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    return tensor

def main():
    parser = argparse.ArgumentParser(description="Test DINO-AugSeg & Save Visuals")

    parser.add_argument("--data_root",   type=str, required=True)
    parser.add_argument("--repo_dir",    type=str, default=".")
    parser.add_argument("--backbone_weight", type=str, required=True)
    parser.add_argument("--ckpt_path",   type=str, required=True)
    
    parser.add_argument("--backbone",    type=str, default="convnext_small", choices=list(_BACKBONE_TABLE.keys()))
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--img_size",    type=int, default=512)
    parser.add_argument("--mask_suffix", type=str, default="")
    parser.add_argument("--images_dir",  type=str, default="images")
    parser.add_argument("--labels_dir",  type=str, default="annotations")
    parser.add_argument("--use_wt_aug",  action="store_true", default=True)
    
    parser.add_argument("--num_support", type=int, default=5)
    parser.add_argument("--num_query",   type=int, default=4)
    parser.add_argument("--episodes",    type=int, default=20, help="Số episodes xuất ảnh (Đừng để quá cao kẻo đầy ổ cứng)")
    parser.add_argument("--num_workers", type=int, default=4)
    
    parser.add_argument("--save_dir",    type=str, default="test_results", help="Thư mục lưu ảnh kết quả")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

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

    # 2. Load Model
    backbone, model_type = load_backbone(backbone=args.backbone, weight_path=args.backbone_weight, repo_dir=args.repo_dir)
    model = DINO_AugSeg(
        encoder=backbone, num_classes=args.num_classes,
        model_type=model_type, use_wt_aug=args.use_wt_aug,
    ).to(device)

    state_dict = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.update_random_choice(0.0) 
    model.eval()

    # 3. Test & Save Loop
    print("\n" + "="*60)
    print(f"  ĐANG TEST VÀ LƯU ẢNH VÀO: {args.save_dir}")
    print("="*60)
    
    # Đồng thời tính metrics
    m = validate_metrics(model, test_loader, device, num_classes=args.num_classes, ignore_background=True)

    print("\nĐang xuất file hình ảnh (Visualizing)...")
    pbar = tqdm(test_loader, desc="Saving Visuals")
    for episode_idx, (s_imgs, s_masks, q_imgs, q_masks) in enumerate(pbar):
        s_imgs = s_imgs.squeeze(0).to(device)
        s_masks = s_masks.squeeze(0).to(device)
        q_imgs = q_imgs.squeeze(0).to(device)
        q_masks = q_masks.squeeze(0).to(device)

        # Rút Prototype (Chunking để tránh OOM)
        protos_list = []
        with torch.no_grad():
            for i in range(0, s_imgs.size(0), 3):
                mini_s = s_imgs[i : i+3]
                mini_m = s_masks[i : i+3]
                mini_p = model.compute_prototype(mini_s, mini_m)
                protos_list.append(mini_p)
        final_prototypes = torch.cat(protos_list, dim=1)
        final_prototypes = F.normalize(final_prototypes, p=2, dim=-1)

        # Suy luận
        with torch.no_grad():
            preds = model(q_imgs, prototypes=final_prototypes)
            pred_cls = torch.softmax(preds, dim=1).argmax(dim=1)

        # Xử lý và lưu từng ảnh Query trong Episode này
        for q_idx in range(q_imgs.size(0)):
            # Phục hồi ảnh gốc RGB
            orig_img_tensor = unnormalize(q_imgs[q_idx])
            orig_img_np = (orig_img_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

            gt_mask = q_masks[q_idx]
            pr_mask = pred_cls[q_idx]

            # Tô màu
            gt_color = colorize_mask(gt_mask, args.num_classes)
            pr_color = colorize_mask(pr_mask, args.num_classes)

            # Overlay Ground Truth (Blended)
            alpha = 0.5
            gt_has_mask = (gt_color.sum(axis=2) > 0)
            blended_gt = orig_img_np.copy()
            blended_gt[gt_has_mask] = orig_img_np[gt_has_mask] * (1 - alpha) + gt_color[gt_has_mask] * alpha

            # Overlay Prediction (Blended)
            pr_has_mask = (pr_color.sum(axis=2) > 0)
            blended_pr = orig_img_np.copy()
            blended_pr[pr_has_mask] = orig_img_np[pr_has_mask] * (1 - alpha) + pr_color[pr_has_mask] * alpha

            # Nối 4 ảnh lại thành 1 dải (Grid): [Gốc | Ground Truth | Kết quả Model | Model đè lên ảnh gốc]
            # Sẽ rất tiện để bạn chèn vào bài báo!
            grid = np.concatenate([orig_img_np, blended_gt, pr_color, blended_pr], axis=1)

            # Lưu ảnh
            filename = os.path.join(args.save_dir, f"ep_{episode_idx:03d}_query_{q_idx:02d}.jpg")
            Image.fromarray(grid).save(filename)

    print(f"\n✅ Đã lưu toàn bộ ảnh tại thư mục: {args.save_dir}")
    print(f"✅ mIoU cuối cùng của đợt test: {m['mean_iou']*100:.2f}%")

if __name__ == "__main__":
    main()
