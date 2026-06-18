"""
CustomDataset — Tập dữ liệu ảnh 2D tổng quát cho DINO-AugSeg
=============================================================
Hỗ trợ cấu trúc thư mục:

  <data_root>/
      Images/
          train/      # ảnh training (JPG, PNG, BMP, ...)
          val/        # ảnh validation
          test/       # ảnh test
      labels/
          train/      # mask tương ứng (cùng tên file, grayscale)
          val/
          test/

Quy ước Mask:
  - Binary (num_classes=2)   : mask grayscale, pixel > 127 → class 1, còn lại → 0
  - Multi-class              : mask grayscale, giá trị pixel = nhãn lớp (0, 1, 2, ...)

Ví dụ sử dụng:
    from dataset.custom_dataset import CustomDataset

    train_ds = CustomDataset(
        data_root="datasets",
        split="train",
        num_classes=2,
        img_size=512,
    )
"""

import os
import random
from glob import glob

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Các định dạng ảnh được hỗ trợ
# ---------------------------------------------------------------------------
_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _find_label_file(label_dir: str, stem: str) -> str:
    """
    Tìm file mask có tên `stem` với bất kỳ extension ảnh nào trong `label_dir`.
    Ưu tiên PNG (thường dùng cho mask), sau đó thử các định dạng khác.
    """
    priority = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    for ext in priority:
        for e in (ext, ext.upper()):
            p = os.path.join(label_dir, stem + e)
            if os.path.isfile(p):
                return p
    raise FileNotFoundError(
        f"Không tìm thấy mask '{stem}' (PNG/JPG/BMP/TIFF) trong '{label_dir}'"
    )


def _scan_image_dir(img_dir: str):
    """
    Quét thư mục `img_dir` và trả về danh sách (stem, full_path) đã sắp xếp.
    Chỉ lấy file có extension trong _IMG_EXTENSIONS.
    """
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(
            f"Không tìm thấy thư mục ảnh: '{img_dir}'\n"
            f"Hãy đảm bảo cấu trúc: <data_root>/Images/<split>/"
        )

    entries = []
    for fname in sorted(os.listdir(img_dir)):
        stem, ext = os.path.splitext(fname)
        if ext.lower() in _IMG_EXTENSIONS:
            entries.append((stem, os.path.join(img_dir, fname)))

    if len(entries) == 0:
        raise ValueError(
            f"Không tìm thấy ảnh nào trong '{img_dir}'. "
            f"Hỗ trợ: {', '.join(_IMG_EXTENSIONS)}"
        )
    return entries


# ---------------------------------------------------------------------------
# CustomDataset — Dataset chính
# ---------------------------------------------------------------------------
class CustomDataset(Dataset):
    """
    Dataset 2D tổng quát cho DINO-AugSeg.

    Cấu trúc thư mục cần có:
        <data_root>/
            Images/
                train/   val/   test/
            labels/
                train/   val/   test/

    Args:
        data_root     : Đường dẫn thư mục gốc chứa Images/ và labels/.
        split         : 'train' | 'val' | 'test'
        num_classes   : Số lớp phân đoạn (bao gồm background).
                        - 2  → binary segmentation
                        - >2 → multi-class
        transform_img : torchvision.transforms áp dụng cho ảnh đầu vào.
                        Nếu None → dùng Resize + ToTensor + Normalize mặc định.
        img_size      : Kích thước resize về H×W (vuông).
        train_num     : Số mẫu train tối đa. "all" = dùng toàn bộ.
        mask_suffix   : Suffix thêm vào stem của mask (ví dụ "_mask").
                        Thường để "" nếu ảnh và mask cùng tên.
        images_dir    : Tên thư mục chứa ảnh (mặc định "Images").
        labels_dir    : Tên thư mục chứa mask (mặc định "labels").
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        num_classes: int = 2,
        transform_img=None,
        img_size: int = 512,
        train_num="all",
        mask_suffix: str = "",
        images_dir: str = "images",
        labels_dir: str = "labels",
    ):
        assert split in ("train", "val", "test"), \
            "split phải là 'train' | 'val' | 'test'"

        self.data_root = data_root
        self.split = split
        self.num_classes = num_classes
        self.img_size = img_size
        self.mask_suffix = mask_suffix
        self.is_binary = (num_classes == 2)

        # Thư mục ảnh và mask theo split
        self.img_dir   = os.path.join(data_root, images_dir, split)
        self.label_dir = os.path.join(data_root, labels_dir,  split)

        if not os.path.isdir(self.label_dir):
            raise FileNotFoundError(
                f"Không tìm thấy thư mục mask: '{self.label_dir}'\n"
                f"Hãy đảm bảo cấu trúc: <data_root>/labels/<split>/"
            )

        # Scan ảnh từ thư mục
        entries = _scan_image_dir(self.img_dir)  # [(stem, path), ...]

        # Giới hạn số mẫu training
        if split == "train" and train_num != "all":
            target_num = int(train_num)
            if target_num < len(entries):
                random.seed(42)  # Giữ cố định tập ảnh được chọn qua các lần chạy
                entries = random.sample(entries, target_num)
                entries.sort(key=lambda x: x[0])

        self.samples = entries  # list of (stem, img_path)
        print(
            f"[CustomDataset] split={split:5s} | {len(self.samples):5d} ảnh "
            f"| num_classes={num_classes} | img_size={img_size}\n"
            f"  Images : {self.img_dir}\n"
            f"  Labels : {self.label_dir}"
        )

        # Transform mặc định nếu không truyền vào
        self.transform_img = transform_img or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        stem, img_path = self.samples[idx]

        # ---- Load ảnh ----
        img = Image.open(img_path).convert("RGB")

        # ---- Load mask ----
        mask_stem = stem + self.mask_suffix
        mask_path = _find_label_file(self.label_dir, mask_stem)
        mask = Image.open(mask_path).convert("L")   # grayscale

        # ---- Augmentation cho training ----
        if self.split == "train":
            img, mask = _random_flip(img, mask)
            img, mask = _random_rotate90(img, mask)
            img, mask = _random_crop_resize(img, mask, self.img_size)

        # ---- Transform ảnh ----
        img_tensor = self.transform_img(img)   # [3, H, W]

        # ---- Xử lý mask ----
        mask = mask.resize((self.img_size, self.img_size), resample=Image.NEAREST)
        mask_np = np.array(mask, dtype=np.uint8)

        if self.is_binary:
            # SỬA Ở ĐÂY: Thay vì > 127, chỉ cần > 0 là thành foreground (1)
            # Bất kể mask là 1, 2 hay 255, cứ lớn hơn 0 là tính làm vật thể.
            mask_np = (mask_np > 0).astype(np.uint8)
        else:
            # Multi-class: Giữ nguyên 0..num_classes-1,
            # mọi giá trị khác (bao gồm 255) → 255 (ignore_index)
            # TRƯỚC ĐÂY: np.clip ép 255 → class 6, gây nhiễu Loss.
            mask_np = np.where(mask_np < self.num_classes, mask_np, 255).astype(np.uint8)

        mask_tensor = torch.from_numpy(mask_np).long()   # [H, W]  # [H, W]

        return img_tensor, mask_tensor


# ---------------------------------------------------------------------------
# Augmentation đơn giản (không cần albumentations)
# ---------------------------------------------------------------------------
def _random_flip(img: Image.Image, mask: Image.Image):
    """Random horizontal/vertical flip."""
    if random.random() > 0.5:
        img  = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() > 0.5:
        img  = img.transpose(Image.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
    return img, mask


def _random_rotate90(img: Image.Image, mask: Image.Image):
    """Random 90-degree rotation (0°, 90°, 180°, 270°)."""
    k = random.choice([0, 1, 2, 3])
    if k > 0:
        angle = k * 90
        img  = img.rotate(angle)
        mask = mask.rotate(angle)
    return img, mask


def _random_crop_resize(img: Image.Image, mask: Image.Image, size: int):
    """
    Random crop (scale 0.7–1.0 của kích thước gốc) rồi resize về `size`.
    Giúp mô hình học được nhiều scale khác nhau.
    """
    w, h = img.size
    scale = random.uniform(0.7, 1.0)
    crop_w = int(w * scale)
    crop_h = int(h * scale)
    x0 = random.randint(0, w - crop_w)
    y0 = random.randint(0, h - crop_h)
    img  = img.crop((x0, y0, x0 + crop_w, y0 + crop_h))
    mask = mask.crop((x0, y0, x0 + crop_w, y0 + crop_h))
    img  = img.resize((size, size), resample=Image.BILINEAR)
    mask = mask.resize((size, size), resample=Image.NEAREST)
    return img, mask


# ---------------------------------------------------------------------------
# Tiện ích: liệt kê thống kê dataset
# ---------------------------------------------------------------------------
def print_dataset_stats(data_root: str,
                        images_dir: str = "images",
                        labels_dir: str = "labels"):
    """In thống kê số lượng ảnh trong từng split."""
    print(f"\n{'='*55}")
    print(f"  Dataset tại: {data_root}")
    print(f"{'='*55}")
    for split in ("train", "val", "test"):
        img_dir   = os.path.join(data_root, images_dir, split)
        label_dir = os.path.join(data_root, labels_dir, split)
        n_img = n_lbl = 0
        if os.path.isdir(img_dir):
            n_img = sum(
                1 for f in os.listdir(img_dir)
                if os.path.splitext(f)[1].lower() in _IMG_EXTENSIONS
            )
        if os.path.isdir(label_dir):
            n_lbl = sum(
                1 for f in os.listdir(label_dir)
                if os.path.splitext(f)[1].lower() in _IMG_EXTENSIONS
            )
        status = "✅" if n_img == n_lbl and n_img > 0 else "⚠️ "
        print(f"  {status} {split:5s}: {n_img:5d} ảnh | {n_lbl:5d} mask")
    print(f"{'='*55}\n")

# ---------------------------------------------------------------------------
# Episodic Dataset (Class-Aware Sampling)
# ---------------------------------------------------------------------------
class EpisodicDataset(Dataset):
    """
    Class bọc (Wrapper) biến CustomDataset gốc thành tập dữ liệu Episodic.
    Đã được nâng cấp lên Class-Aware Sampling để trị các class cực hiếm.
    """
    def __init__(self, base_dataset: CustomDataset, num_support=15, num_query=1, episodes_per_epoch=500, deterministic=False):
        self.base = base_dataset
        self.num_support = num_support
        self.num_query = num_query
        self.episodes_per_epoch = episodes_per_epoch
        self.deterministic = deterministic  # True cho Val/Test để kết quả lặp lại được
        
        print(f"[{self.base.split.upper()}] Đang quét toàn bộ Mask để xây dựng Class-Aware Sampler (Khoảng 5-15 giây)...")
        from PIL import Image
        import numpy as np
        
        # Bỏ qua background (0) và vùng ignore (255)
        self.class_to_indices = {c: [] for c in range(1, self.base.num_classes)}
        
        for i, (stem, img_path) in enumerate(self.base.samples):
            mask_stem = stem + self.base.mask_suffix
            try:
                mask_path = _find_label_file(self.base.label_dir, mask_stem)
                mask = Image.open(mask_path).convert("L")
                mask_np = np.array(mask)
                unique_classes = np.unique(mask_np)
                for c in unique_classes:
                    if 0 < c < self.base.num_classes:
                        self.class_to_indices[c].append(i)
            except Exception as e:
                pass # Bỏ qua nếu lỗi load mask
                
        # Lọc bỏ những class không có ảnh nào
        self.available_classes = [c for c, idxs in self.class_to_indices.items() if len(idxs) > 0]
        print(f"Xây dựng xong Sampler! Các class hợp lệ có thể bốc thăm: {self.available_classes}")

    def __len__(self):
        return self.episodes_per_epoch

    def __getitem__(self, idx):
        # Khi deterministic=True (Val/Test): dùng RNG cục bộ seed theo idx
        # → cùng idx luôn cho ra cùng episode → kết quả Validation ổn định.
        # Khi deterministic=False (Train): dùng RNG toàn cục → ngẫu nhiên mỗi lần.
        if self.deterministic:
            rng = random.Random(idx + 9999)
        else:
            rng = random

        supp_idx_set = set()
        
        # 1. Bốc thăm có chủ đích (Ép các class phải xuất hiện)
        chosen_classes = []
        if len(self.available_classes) > 0:
            k_classes = min(self.num_support, len(self.available_classes))
            chosen_classes = rng.sample(self.available_classes, k=k_classes)
            
            for c in chosen_classes:
                chosen_idx = rng.choice(self.class_to_indices[c])
                supp_idx_set.add(chosen_idx)
                
        # 2. Bốc ngẫu nhiên thêm cho đủ num_support
        supp_idx = list(supp_idx_set)
        while len(supp_idx) < self.num_support:
            candidate = rng.randint(0, len(self.base) - 1)
            if candidate not in supp_idx:
                supp_idx.append(candidate)
                
        # 3. Bốc Query Set — ĐỒNG BỘ VỚI SUPPORT SET
        query_idx = []
        query_idx_set = set()
        
        if len(chosen_classes) > 0:
            # Bước 3a: Ép mỗi chosen_class có ít nhất 1 ảnh trong Query
            classes_for_query = list(chosen_classes)
            rng.shuffle(classes_for_query)
            
            for c in classes_for_query:
                if len(query_idx) >= self.num_query:
                    break
                candidates = [i for i in self.class_to_indices[c] 
                              if i not in supp_idx and i not in query_idx_set]
                if len(candidates) > 0:
                    chosen = rng.choice(candidates)
                    query_idx.append(chosen)
                    query_idx_set.add(chosen)
        
        # Bước 3b: Bốc thêm từ pool các class đã chọn
        if len(query_idx) < self.num_query and len(chosen_classes) > 0:
            class_aware_pool = []
            for c in chosen_classes:
                class_aware_pool.extend(self.class_to_indices[c])
            class_aware_pool = list(set(class_aware_pool))
            rng.shuffle(class_aware_pool)
            
            for candidate in class_aware_pool:
                if len(query_idx) >= self.num_query:
                    break
                if candidate not in supp_idx and candidate not in query_idx_set:
                    query_idx.append(candidate)
                    query_idx_set.add(candidate)
        
        # Bước 3c: Fallback
        while len(query_idx) < self.num_query:
            candidate = rng.randint(0, len(self.base) - 1)
            if candidate not in supp_idx and candidate not in query_idx_set:
                query_idx.append(candidate)
                query_idx_set.add(candidate)

        # 4. Load Tensor
        s_imgs, s_masks = [], []
        for i in supp_idx:
            img, mask = self.base[i]
            s_imgs.append(img)
            s_masks.append(mask)

        q_imgs, q_masks = [], []
        for i in query_idx:
            img, mask = self.base[i]
            q_imgs.append(img)
            q_masks.append(mask)

        return torch.stack(s_imgs), torch.stack(s_masks), torch.stack(q_imgs), torch.stack(q_masks)
# ---------------------------------------------------------------------------
# Test nhanh
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Kiểm tra CustomDataset với cấu trúc Images/ + labels/"
    )
    parser.add_argument("--data_root",   type=str, required=True,
                        help="Thư mục gốc (chứa Images/ và labels/)")
    parser.add_argument("--split",       type=str, default="train",
                        choices=["train", "val", "test"])
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--img_size",    type=int, default=512)
    parser.add_argument("--images_dir",  type=str, default="images")
    parser.add_argument("--labels_dir",  type=str, default="annotations")
    parser.add_argument("--stats",       action="store_true",
                        help="Chỉ in thống kê số lượng ảnh, không load dataset")
    args = parser.parse_args()

    if args.stats:
        print_dataset_stats(args.data_root, args.images_dir, args.labels_dir)
    else:
        print_dataset_stats(args.data_root, args.images_dir, args.labels_dir)

        ds = CustomDataset(
            data_root=args.data_root,
            split=args.split,
            num_classes=args.num_classes,
            img_size=args.img_size,
            images_dir=args.images_dir,
            labels_dir=args.labels_dir,
        )
        img, mask = ds[10]
        print(f"\n--- Sample[0] ---")
        print(f"Image shape : {img.shape}   dtype={img.dtype}")
        print(f"Mask  shape : {mask.shape}  dtype={mask.dtype}")
        print(f"Mask  unique: {mask.unique().tolist()}")
        print(f"Mask  min/max: {mask.min().item()} / {mask.max().item()}")
