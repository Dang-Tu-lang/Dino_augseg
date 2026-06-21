import os
import argparse
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import cv2

from model.Dinov3_WTAUG_UNet import DINO_AugSeg
from train_custom import load_backbone, _BACKBONE_TABLE

# Bảng màu cho 7 classes (Background + 6 loại san hô)
# Bạn có thể tự chỉnh mã RGB cho vừa mắt. Background (0) là màu Đen.
COLOR_PALETTE = [
    [0, 0, 0],       # Class 0: Background (Không tô màu)
    [255, 0, 0],     # Class 1: Đỏ rực
    [0, 255, 0],     # Class 2: Xanh lá cây
    [0, 150, 255],   # Class 3: Xanh nước biển
    [255, 255, 0],   # Class 4: Vàng
    [255, 0, 255],   # Class 5: Tím / Hồng
    [0, 255, 255],   # Class 6: Cyan (Xanh lơ)
]

def colorize_mask(mask_tensor, num_classes=7):
    """Biến đổi ma trận nhãn [H, W] thành ma trận RGB [H, W, 3]"""
    mask_np = mask_tensor.cpu().numpy()
    h, w = mask_np.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(1, num_classes): # Bỏ qua background (c=0)
        color_mask[mask_np == c] = COLOR_PALETTE[c]
    return color_mask

def main():
    parser = argparse.ArgumentParser(description="Visual Inference DINO-AugSeg (Dành cho 1 ảnh thực tế)")

    # Tham số mô hình
    parser.add_argument("--repo_dir",    type=str, default=".", help="Thư mục repo DINO-AugSeg")
    parser.add_argument("--backbone_weight", type=str, required=True, help="Đường dẫn đến backbone .pth")
    parser.add_argument("--ckpt_path",   type=str, required=True, help="Đường dẫn đến model .pth (VD: best_iou_model.pth)")
    parser.add_argument("--backbone",    type=str, default="convnext_small", choices=list(_BACKBONE_TABLE.keys()))
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--img_size",    type=int, default=512)
    parser.add_argument("--use_wt_aug",  action="store_true", default=True)
    
    # Tham số dữ liệu đầu vào
    parser.add_argument("--s_img_dir",   type=str, required=True, help="Thư mục chứa 5 ảnh Support")
    parser.add_argument("--s_mask_dir",  type=str, required=True, help="Thư mục chứa 5 nhãn Support tương ứng")
    parser.add_argument("--q_img",       type=str, required=True, help="Đường dẫn đến Bức ảnh Thực tế (Query) cần dự đoán")
    parser.add_argument("--out_path",    type=str, default="infer_result.jpg", help="Nơi lưu ảnh đã tô màu kết quả")
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Transform function (Giống hệ thống Training)
    transform_img = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # 2. Đọc Support Set
    print("\n[1] Đang nạp Support Set...")
    s_imgs, s_masks = [], []
    img_files = sorted([f for f in os.listdir(args.s_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    mask_files = sorted([f for f in os.listdir(args.s_mask_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    if len(img_files) == 0 or len(img_files) != len(mask_files):
        print(f"❌ Lỗi: Số lượng ảnh và mask support không khớp hoặc bằng 0 ({len(img_files)} vs {len(mask_files)})")
        return

    for img_name, mask_name in zip(img_files, mask_files):
        # Đọc và biến đổi ảnh
        img_path = os.path.join(args.s_img_dir, img_name)
        img_pil = Image.open(img_path).convert("RGB")
        img_tensor = transform_img(img_pil)
        s_imgs.append(img_tensor)
        
        # Đọc và biến đổi mask (Nearest để không bị nhòe label)
        mask_path = os.path.join(args.s_mask_dir, mask_name)
        mask_pil = Image.open(mask_path)
        mask_pil = mask_pil.resize((args.img_size, args.img_size), Image.NEAREST)
        mask_np = np.array(mask_pil)
        s_masks.append(torch.from_numpy(mask_np).long())

    # Gộp thành Tensor Batch: [N, C, H, W] và [N, H, W]
    s_imgs = torch.stack(s_imgs).to(device)
    s_masks = torch.stack(s_masks).to(device)
    print(f"    -> Đã nạp thành công {s_imgs.size(0)} ảnh Support.")

    # 3. Khởi tạo Mô hình
    print("\n[2] Đang khởi tạo DINO-AugSeg...")
    backbone, model_type = load_backbone(backbone=args.backbone, weight_path=args.backbone_weight, repo_dir=args.repo_dir)
    model = DINO_AugSeg(
        encoder=backbone, num_classes=args.num_classes,
        model_type=model_type, use_wt_aug=args.use_wt_aug,
    ).to(device)

    print(f"    -> Đang nạp Tri thức từ Checkpoint: {args.ckpt_path}")
    state_dict = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # Tắt hoàn toàn cơ chế Random Mask (chỉ dùng lúc Train) để hình ảnh ổn định nhất
    model.update_random_choice(0.0) 
    model.eval()

    # 4. Trích xuất Prototype từ Support Set
    print("\n[3] Đang trích xuất đặc trưng từ Support Set (Dense Matching)...")
    with torch.no_grad():
        support_features, support_masks_r = model.extract_support_features(s_imgs, s_masks)
    
    # 5. Xử lý ảnh Thực tế (Query Image)
    print(f"\n[4] Đang phóng sóng Sonar (Inference) vào ảnh: {args.q_img}")
    q_pil_origin = Image.open(args.q_img).convert("RGB")
    q_w, q_h = q_pil_origin.size # Giữ lại kích thước gốc để lúc lưu ảnh không bị méo
    
    q_tensor = transform_img(q_pil_origin).unsqueeze(0).to(device) # Cấp chiều Batch = 1 -> [1, C, H, W]

    with torch.no_grad():
        preds = model(q_tensor, support_features=support_features, support_masks=support_masks_r)
        pred_cls = torch.softmax(preds, dim=1).argmax(dim=1).squeeze(0) # Trả về tọa độ [H, W]

    # 6. Tô màu và Kết xuất
    print("\n[5] Đang vẽ lại bản đồ San hô...")
    color_mask = colorize_mask(pred_cls, args.num_classes)
    
    # Phóng to lớp màu về lại kích thước ảnh gốc của camera
    color_mask_resized = cv2.resize(color_mask, (q_w, q_h), interpolation=cv2.INTER_NEAREST)
    
    # Trộn màu (Alpha Blending)
    alpha = 0.5 # Độ trong suốt của lớp màu (0.0 -> gốc, 1.0 -> màu đặc)
    origin_np = np.array(q_pil_origin)
    
    has_mask = (color_mask_resized.sum(axis=2) > 0) # Những vùng nào được mô hình nhận diện là san hô
    
    blended = origin_np.copy()
    blended[has_mask] = origin_np[has_mask] * (1 - alpha) + color_mask_resized[has_mask] * alpha

    # Lưu file
    blended_img = Image.fromarray(blended.astype(np.uint8))
    blended_img.save(args.out_path)
    
    print(f"✅ HOÀN TẤT! Ảnh dự đoán đã được lưu tại: {args.out_path}")

if __name__ == "__main__":
    main()
