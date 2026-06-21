import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import random
from pytorch_wavelets import DWTForward, DWTInverse
from typing import Tuple, Optional

"""
Please check:
1. class DINO_AugSeg(nn.Module): build the DINO-AugSeg
2. class AttentionCrossDecoder_WT_ALL(nn.Module): include CG-Fuse and WT-Aug of the paper
    1> CG-Fuse: contextual-guided feature fusion module to leverage the high-level contextual information from DINOv3 feature
    2> WT-Aug (only used in training): wavelet-based feature-level augmentation method for DINOv3 features
3. The paper could be download in : https://www.arxiv.org/abs/2601.08078
"""

class ImageAugmentor(nn.Module):
    def __init__(self, prob=0.7, drop_rate=0.1):
        super().__init__()
        self.prob = prob
        self.drop_rate = drop_rate

    # ----------- augmentation ops ----------------
    def aug_brightness(self, x):
        """Random brightness scaling."""
        scale = torch.empty(x.size(0), 1, 1, 1, device=x.device).uniform_(0.7, 1.3)
        return x * scale

    def aug_motion(self, x, kernel=7):
        """Horizontal motion blur."""
        b, c, h, w = x.shape
        k = torch.ones((c, 1, 1, kernel), device=x.device) / kernel
        return F.conv2d(x, k, padding=(0, kernel // 2), groups=c)

    def aug_poisson(self, x):
        """Poisson noise."""
        x_pos = torch.clamp(x, min=0)
        return torch.poisson(x_pos)

    def aug_random_zero(self, x):
        """Randomly mask pixels."""
        mask = (torch.rand_like(x) > self.drop_rate).float()
        return x * mask

    # ----------- forward ----------------
    def forward(self, x):  # x: B×C×H×W
        if random.random() > self.prob:
            return x
        # if not self.training:
        #     return x

        aug_type = random.choice(["brightness", "motion", "poisson", "random_zero"])

        if aug_type == "brightness":
            return self.aug_brightness(x)
        elif aug_type == "motion":
            return self.aug_motion(x)
        elif aug_type == "poisson":
            return self.aug_poisson(x)
        elif aug_type == "random_zero":
            return self.aug_random_zero(x)

        return x


## feature augmentation module
class FeatureAugmentor(nn.Module):
    def __init__(self, prob=0.7, drop_rate=0.2):
        super().__init__()
        self.prob = prob
        self.drop_rate = drop_rate

    def forward(self, feats):
        f_list = list(feats)

        if random.random() > self.prob:
            return feats

        # randomly choose one feature level to augment
        idx = random.randint(0, len(f_list) - 1)
        feat = f_list[idx]

        # randomly choose augmentation method
        aug_type = random.choice(["brightness", "motion", "poisson", "random_zero"])

        if aug_type == "brightness":
            feat = self.brightness_aug(feat)

        elif aug_type == "motion":
            feat = self.motion_blur(feat)

        elif aug_type == "poisson":
            feat = self.poisson_noise(feat)

        elif aug_type == "random_zero":
            feat = self.random_zero_mask(feat)

        f_list[idx] = feat
        return tuple(f_list)

    # -------------------------------------
    # Brightness
    # -------------------------------------
    def brightness_aug(self, feat):
        alpha = random.uniform(0.7, 1.3)
        return feat * alpha

    # -------------------------------------
    # Motion Blur
    # -------------------------------------
    def motion_blur(self, feat):
        kernel_size = 7
        kernel = torch.zeros((kernel_size, kernel_size))
        kernel[kernel_size//2, :] = 1.0
        kernel /= kernel.sum()

        kernel = kernel.unsqueeze(0).unsqueeze(0).to(feat.device)
        B, C, H, W = feat.shape
        kernel = kernel.repeat(C, 1, 1, 1)

        return F.conv2d(feat, kernel, padding=kernel_size//2, groups=C)

    # -------------------------------------
    # Poisson Noise
    # -------------------------------------
    def poisson_noise(self, feat):
        min_val = feat.min()
        shifted = feat - min_val   # ensure >= 0

        scale = 20
        shifted = shifted * scale

        # torch.poisson only runs on CPU
        noisy = torch.poisson(shifted.cpu()).to(feat.device)

        noisy = noisy / scale
        noisy = noisy + min_val

        return noisy

    # -------------------------------------
    # Random Zero Masking (Dropout-style)
    # -------------------------------------
    def random_zero_mask(self, feat):
        """
        Randomly sets positions to zero, similar to your example:
        mask = (rand > drop_rate)
        """
        mask = (torch.rand_like(feat) > self.drop_rate).float()
        return feat * mask


# -----------------------
# Helpers: window partition / reverse (Swin-style)
# -----------------------
def window_partition(x: torch.Tensor, window_size: Tuple[int,int]) -> torch.Tensor:
    """
    x: (B, H, W, C)
    return: (num_windows*B, Wh*Ww, C)
    """
    B, H, W, C = x.shape
    Wh, Ww = window_size
    x = x.view(B, H // Wh, Wh, W // Ww, Ww, C)
    x = x.permute(0,1,3,2,4,5).contiguous()
    windows = x.view(-1, Wh * Ww, C)
    return windows

def window_reverse(windows: torch.Tensor, window_size: Tuple[int,int], H: int, W: int) -> torch.Tensor:
    """
    windows: (num_windows*B, Wh*Ww, C)
    return: (B, H, W, C)
    """
    Wh, Ww = window_size
    B = int(windows.shape[0] // (H // Wh * W // Ww))
    x = windows.view(B, H // Wh, W // Ww, Wh, Ww, -1)
    x = x.permute(0,1,3,2,4,5).contiguous()
    x = x.view(B, H, W, -1)
    return x

# -----------------------
# RoPE utilities
# -----------------------
def build_rope_cache(seq_len: int, dim_head: int, base: int = 10000, device=None) -> torch.Tensor:
    """Return (1, seq_len, dim_head) containing [sin, cos] concat (sin first then cos),
       encoded as (seq_len, dim_head) where dim_head is even and output has shape (1, seq_len, dim_head).
       We'll pack as [sin, cos] interleaved when applying.
    """
    pos = torch.arange(seq_len, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim_head, 2, device=device).float() / dim_head))
    sinusoid = torch.einsum("i,j->ij", pos, inv_freq)  # (seq_len, dim_head/2)
    sin = sinusoid.sin()
    cos = sinusoid.cos()
    # Interleave sin and cos into dim_head: [sin0, cos0, sin1, cos1, ...]
    sin_cos = torch.stack([sin, cos], dim=-1).reshape(seq_len, dim_head)
    return sin_cos.unsqueeze(0)  # (1, seq_len, dim_head)

def apply_rope_tensor(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    """
    x: (B, nH, N, d) where d = dim_head
    rope_cache: (1, N, d) where d is even and is arranged [sin0,cos0,sin1,cos1,...]
    Implementation uses rotation: for each pair (x_2i, x_2i+1):
      [x_2i', x_2i+1'] = [ x_2i * cos - x_2i+1 * sin, x_2i * sin + x_2i+1 * cos ]
    """
    # x and rope_cache broadcast over batch and heads
    # reshape last dim to (d/2, 2) to vectorize
    B, nH, N, d = x.shape
    assert d % 2 == 0, "head dim must be even for RoPE"
    x_ = x.view(B, nH, N, d // 2, 2)     # (B, nH, N, d/2, 2)
    rope = rope_cache.view(1, N, d // 2, 2).to(x.device)  # (1, N, d/2, 2)
    sin = rope[..., 0]  # (1, N, d/2)
    cos = rope[..., 1]  # (1, N, d/2)
    sin = sin.unsqueeze(0).unsqueeze(1)  # (1,1,N,d/2)
    cos = cos.unsqueeze(0).unsqueeze(1)
    x0 = x_[..., 0]  # (B,nH,N,d/2)
    x1 = x_[..., 1]
    xr0 = x0 * cos - x1 * sin
    xr1 = x0 * sin + x1 * cos
    xr = torch.stack([xr0, xr1], dim=-1).view(B, nH, N, d)
    return xr

# -----------------------
# Relative position bias for windows (Swin-style)
# -----------------------
class RelativePositionBias(nn.Module):
    def __init__(self, window_size: Tuple[int,int], num_heads: int):
        super().__init__()
        Wh, Ww = window_size
        self.window_size = window_size
        self.num_heads = num_heads
        # number of relative positions
        table_size = (2 * Wh - 1) * (2 * Ww - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(table_size, num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # create index
        coords_h = torch.arange(Wh)
        coords_w = torch.arange(Ww)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[..., 0] += Wh - 1
        relative_coords[..., 1] += Ww - 1
        relative_coords[..., 0] *= 2 * Ww - 1
        relative_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_index", relative_index)

    def forward(self) -> torch.Tensor:
        # returns (num_heads, Wh*Ww, Wh*Ww)
        bias = self.relative_position_bias_table[self.relative_index.view(-1)].view(
            self.relative_index.shape[0], self.relative_index.shape[1], -1
        )  # Wh*Ww, Wh*Ww, num_heads
        bias = bias.permute(2, 0, 1).contiguous()
        return bias  # (num_heads, N, N)

class ParameterFreeDenseGate(nn.Module):
    def __init__(self, temperature=10.0):
        """
        temperature: Hệ số khuếch đại. 
        - Giá trị càng cao (10-20), cổng đóng/mở càng sắc nét (gần về 0 hoặc 1).
        - Giá trị thấp (1-5), cổng sẽ mềm hơn (nằm lơ lửng ở 0.5).
        """
        super().__init__()
        self.temperature = temperature
        # Hoàn toàn KHÔNG CÓ nn.Parameter hay nn.Conv2d nào ở đây để bảo vệ Few-shot.

    def forward(self, x, context):
        """
        x: Luồng đặc trưng chính (Query), shape: [B, C, H, W]
        context: Luồng đặc trưng ngữ cảnh từ DINO (Key), shape: [B, C, H_c, W_c]
        """
        B, C, H, W = x.shape
        B_c, C_c, H_c, W_c = context.shape
        
        # 1. Ép phẳng không gian 2D thành chuỗi 1D để nhân ma trận
        # x_flat: [B, H*W, C]
        x_flat = x.view(B, C, -1).transpose(1, 2)         
        # context_flat: [B, H_c*W_c, C]
        context_flat = context.view(B_c, C_c, -1).transpose(1, 2) 

        # 2. Chuẩn hóa L2 (Bắt buộc để tính Cosine Similarity chuẩn xác)
        x_norm = F.normalize(x_flat, p=2, dim=-1)
        context_norm = F.normalize(context_flat, p=2, dim=-1)

        # 3. Nhân ma trận để tính độ tương quan toàn cục (Dense Correlation)
        # Mỗi pixel của x sẽ so sánh với TOÀN BỘ pixel của context
        # affinity shape: [B, H*W, H_c*W_c]
        affinity = torch.matmul(x_norm, context_norm.transpose(-1, -2))

        # 4. Tìm "đối tác" khớp nhất cho mỗi pixel của x
        # max_affinity shape: [B, H*W]
        max_affinity, _ = torch.max(affinity, dim=-1)

        # 5. Kích hoạt Cổng (Sigmoid) + Khuếch đại (Temperature)
        # Giá trị max_affinity nằm trong [-1, 1], qua sigmoid sẽ ép về (0, 1)
        gate_flat = torch.sigmoid(self.temperature * max_affinity)

        # 6. Reshape ngược lại về định dạng ảnh 2D ban đầu
        # gate shape: [B, 1, H, W]
        gate = gate_flat.view(B, 1, H, W)

        return gate

# -----------------------
# Cross Attention Block (global / window / pooled_kv) with RoPE + optional pre-norm
# -----------------------
class CrossAttentionBlock_RoPE(nn.Module):
    def __init__(self,
                 dim_q: int,
                 dim_kv: int,
                 num_heads: int = 8,
                 attn_type: str = "global",   # "global" | "window" | "pooled_kv"
                 window_size: Tuple[int,int] = (7,7),
                 pool_kv: bool = False,
                 pool_stride: int = 2,
                 use_rope: bool = True,
                 pre_norm: bool = True,
                 attn_dropout: float = 0.0,
                 proj_dropout: float = 0.0,
                 use_rel_pos_bias: bool = True):
        super().__init__()
        assert dim_q % num_heads == 0, "dim_q must be divisible by num_heads"
        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.num_heads = num_heads
        self.attn_type = attn_type
        self.window_size = window_size
        self.pool_kv = pool_kv or (attn_type == "pooled_kv")
        self.pool_stride = pool_stride
        self.use_rope = use_rope
        self.pre_norm = pre_norm
        self.use_rel_pos_bias = use_rel_pos_bias and attn_type == "window"

        head_dim = dim_q // num_heads
        self.scale = head_dim ** -0.5

        # projections operate on last dim (feature dim)
        self.q_proj = nn.Linear(dim_q, dim_q)
        self.k_proj = nn.Linear(dim_kv, dim_q)
        self.v_proj = nn.Linear(dim_kv, dim_q)

        self.attn_drop = nn.Dropout(attn_dropout)
        self.out_proj = nn.Linear(dim_q, dim_q)
        self.proj_drop = nn.Dropout(proj_dropout)

        self.gate = ParameterFreeDenseGate(temperature=10.0)

        if pre_norm:
            self.norm_q = nn.LayerNorm(dim_q)
            self.norm_kv = nn.LayerNorm(dim_kv)
        else:
            self.post_norm = nn.LayerNorm(dim_q)

        # window relative bias
        if self.use_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size, num_heads)
        else:
            self.rel_pos_bias = None

        # RoPE cache
        if self.use_rope:
            self._rope_cache: Optional[torch.Tensor] = None
            self._rope_len = 0
            self.head_dim = head_dim

        # pooling
        if self.pool_kv:
            self.pool = nn.AvgPool2d(kernel_size=pool_stride, stride=pool_stride)

    def _maybe_pool_kv(self, kv: torch.Tensor) -> torch.Tensor:
        if self.pool_kv:
            return self.pool(kv)
        return kv

    def _apply_rope_to_QK(self, Q: torch.Tensor, K: torch.Tensor, Nq: int, Nk: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        # Q, K shapes: (B, nH, N, d)
        max_len = max(Nq, Nk)
        if self._rope_cache is None or self._rope_len < max_len or self._rope_cache.device != device:
            self._rope_cache = build_rope_cache(max_len, self.head_dim, device=device)  # (1, max_len, d)
            self._rope_len = max_len
        rope = self._rope_cache  # (1, max_len, d)
        # apply appropriate slices
        Q = apply_rope_tensor(Q, rope[:, :Nq, :].to(device))
        K = apply_rope_tensor(K, rope[:, :Nk, :].to(device))
        return Q, K

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        q: (B, Cq, Hq, Wq)
        kv: (B, Ck, Hk, Wk)
        returns: (B, Cq, Hq, Wq) after cross-attention
        """
        B, Cq, Hq, Wq = q.shape
        Bk, Ck, Hk, Wk = kv.shape
        assert B == Bk, "batch size mismatch"

        # optionally pool kv spatially to reduce compute
        kv_proc = self._maybe_pool_kv(kv)  # shape may change

        # flatten to sequences
        q_seq = q.flatten(2).transpose(1,2)          # (B, Nq, Cq)
        kv_seq = kv_proc.flatten(2).transpose(1,2)   # (B, Nk, Ck)
        Nq = q_seq.shape[1]
        Nk = kv_seq.shape[1]

        # pre-norm
        if self.pre_norm:
            q_seq = self.norm_q(q_seq)
            kv_seq = self.norm_kv(kv_seq)

        # project
        Q = self.q_proj(q_seq)   # (B, Nq, Cq)
        K = self.k_proj(kv_seq)  # (B, Nk, Cq)
        V = self.v_proj(kv_seq)  # (B, Nk, Cq)

        # reshape to (B, nH, N, d)
        d = Cq // self.num_heads
        Q = Q.view(B, Nq, self.num_heads, d).permute(0,2,1,3)  # (B, nH, Nq, d)
        K = K.view(B, Nk, self.num_heads, d).permute(0,2,1,3)  # (B, nH, Nk, d)
        V = V.view(B, Nk, self.num_heads, d).permute(0,2,1,3)  # (B, nH, Nk, d)

        # optionally apply RoPE
        if self.use_rope:
            Q, K = self._apply_rope_to_QK(Q, K, Nq, Nk, q.device)

        # If windowed attention, partition into windows (requires Hq,Wq multiple of window size)
        if self.attn_type == "window":
            Wh, Ww = self.window_size
            assert Hq % Wh == 0 and Wq % Ww == 0, "Hq and Wq must be divisible by window size for windowed attention"
            # convert Q,K,V from (B,nH,N,d) -> (B, nH, Hq, Wq, d) -> partition windows -> (num_windows*B, nH, Wh*Ww, d)
            # easier: reshape Q,K,V to (B, nH, Hq, Wq, d)
            Q_hw = Q.permute(0,1,2,3).contiguous().view(B, self.num_heads, Hq, Wq, d).permute(0,2,3,1,4).contiguous()  # (B,Hq,Wq,nH,d)
            K_hw = K.permute(0,1,2,3).contiguous().view(B, self.num_heads, Hq, Wq, d).permute(0,2,3,1,4).contiguous()
            V_hw = V.permute(0,1,2,3).contiguous().view(B, self.num_heads, Hq, Wq, d).permute(0,2,3,1,4).contiguous()

            # Now for each head, partition windows and compute attention per window independently
            # We'll merge head and batch dims for partition
            # Q_windows: (num_windows*B*nH, Wh*Ww, d)
            Q_windows = []
            K_windows = []
            V_windows = []
            for h_idx in range(self.num_heads):
                Q_h = Q_hw[..., h_idx, :].contiguous()  # (B,Hq,Wq,d)
                K_h = K_hw[..., h_idx, :].contiguous()
                V_h = V_hw[..., h_idx, :].contiguous()
                Qw = window_partition(Q_h.permute(0,2,1,3).contiguous(), (Wh, Ww))  # trick: windows assume (B,H,W,C), use permute to match
                Kw = window_partition(K_h.permute(0,2,1,3).contiguous(), (Wh, Ww))
                Vw = window_partition(V_h.permute(0,2,1,3).contiguous(), (Wh, Ww))
                Q_windows.append(Qw)  # list of (num_windows*B, Wh*Ww, d)
                K_windows.append(Kw)
                V_windows.append(Vw)
            # Stack heads along batch axis: shape -> (nH*(num_windows*B), Wh*Ww, d)
            Qw_cat = torch.cat(Q_windows, dim=0)
            Kw_cat = torch.cat(K_windows, dim=0)
            Vw_cat = torch.cat(V_windows, dim=0)

            # reshape for attention: treat concatenated heads as separate batches
            # compute attention
            q_ = Qw_cat.unsqueeze(2)  # (B', L, 1, d) not necessary
            attn = (Qw_cat @ Kw_cat.transpose(-2,-1)) * self.scale  # (B', L, L)

            # add relative pos bias if available
            if self.rel_pos_bias is not None:
                # rel_pos_bias: (nH, L, L). We need to tile to match concatenated heads ordering
                bias = self.rel_pos_bias()  # (nH, L, L)
                # repeat for batch: each head group corresponds to many windows. We tile bias along batch of windows.
                num_windows_per_image = (Hq // Wh) * (Wq // Ww)
                bias_rep = bias.repeat_interleave(repeats=B * num_windows_per_image, dim=0)  # (nH * num_windows*B, L, L)
                attn = attn + bias_rep.to(attn.device)

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            out_windows = attn @ Vw_cat  # (B', L, d)

            # Now split and merge back per head
            # First, split by heads
            per_head_windows = torch.split(out_windows, out_windows.shape[0] // self.num_heads, dim=0)
            # reconstruct per-head images and then concatenate heads
            head_outputs = []
            for h_idx in range(self.num_heads):
                out_h = per_head_windows[h_idx]  # (num_windows*B, L, d)
                # reverse windows -> (B, Hq, Wq, d)
                out_h_img = window_reverse(out_h, (Wh, Ww), Hq, Wq)  # (B, Hq, Wq, d)
                head_outputs.append(out_h_img)  # list of (B,Hq,Wq,d)
            # stack heads -> (B,Hq,Wq,nH,d) then permute to (B,nH,N,d)
            stacked = torch.stack(head_outputs, dim=3)  # (B,Hq,Wq,nH,d)
            stacked = stacked.view(B, Hq*Wq, self.num_heads, d).permute(0,2,1,3).contiguous()  # (B,nH,N,d)
            # Now merge heads: (B,N,nH,d) -> (B,N,nH*d)
            out = stacked.permute(0,2,1,3).contiguous().view(B, Nq, self.num_heads * d)
            out = self.out_proj(out)
            out = self.proj_drop(out)

            if self.pre_norm:
                # residual q_seq was normalized earlier; add residual and return
                out = out + q_seq
                out = out  # if you want, apply further post-norm variant outside
            if not self.pre_norm:
                out = out + q_seq
                out = self.post_norm(out)

            attn_out_2d = out.transpose(1,2).reshape(B, Cq, Hq, Wq)
            gate_matrix = self.gate(q, kv)
            out = q + gate_matrix * attn_out_2d
            return out

        # ---- Global / pooled_kv attention path ----
        # attention on sequences Q (B,nH,Nq,d), K (B,nH,Nk,d)
        attn = (Q @ K.transpose(-2,-1)) * self.scale  # (B,nH,Nq,Nk)

        # No relative pos bias for global by default (could be added)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ V).permute(0,2,1,3).contiguous().view(B, Nq, self.num_heads * d)  # (B,Nq,Cq)
        out = self.out_proj(out)
        out = self.proj_drop(out)

        # Residual & post-norm
        if self.pre_norm:
            out = out + q_seq
        else:
            out = out + q_seq
            out = self.post_norm(out)

        out = out.transpose(1,2).reshape(B, Cq, Hq, Wq)
        # gate_matrix = self.gate(q, kv)
        # out = q + (q+gate_matrix) * out
        return out

# -------------------------
# helper conv block
# -------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)
##Gate
class GatedConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # Mở rộng channel ra gấp đôi để làm cổng gate
        self.conv_gate = nn.Conv2d(in_ch, out_ch * 2, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch * 2)
        self.act = nn.SiLU(inplace=True)
        # Lớp chiếu cuối cùng để trả về số channel mong muốn nếu cần
        self.proj = nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.bn(self.conv_gate(x))
        # Chia đôi theo chiều channel: một nửa làm đặc trưng, một nửa làm cổng
        feat, gate = torch.chunk(x, 2, dim=1)
        # Cơ chế Gated: Lọc thông tin sau attention
        out = feat * self.act(gate)
        return self.proj(out)

###
##
class LayerNorm2d(nn.Module):
    """LayerNorm chuẩn hóa theo chiều channel cho dữ liệu ảnh (B, C, H, W)"""
    def __init__(self, num_features, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class PostAttentionRefinement(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.ln = LayerNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        
        # Nhánh shortcut nếu kích thước channel thay đổi
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        # Lưu giữ lại identity tensor
        identity = self.shortcut(x)
        
        # Luồng tính toán chính (Pre-LN style)
        out = self.ln(x)
        out = self.act(self.conv1(out))
        out = self.conv2(out)
        
        # Cộng residual giúp ổn định đặc trưng sau Attention
        return out + identity
###


##
class PostAttentionSwiGLU(nn.Module):
    def __init__(self, in_ch, out_ch, hidden_ratio=2):
        super().__init__()
        # 1. Thành phần chuẩn hóa đầu vào (LayerNorm từ Phương án 2)
        self.ln = LayerNorm2d(in_ch)
        
        # Xác định số channel mở rộng ở không gian ẩn của SwiGLU
        hidden_ch = int(in_ch * hidden_ratio)
        
        # 2. Hai nhánh song song của SwiGLU
        self.w_feat = nn.Conv2d(in_ch, hidden_ch, kernel_size=1, padding=0, bias=False)
        self.w_gate = nn.Conv2d(in_ch, hidden_ch, kernel_size=1, padding=0, bias=False)
        
        # Hàm kích hoạt dùng riêng cho nhánh Gate
        self.act = nn.SiLU(inplace=True)
        
        # Lớp chiếu đầu ra (Projection) sau khi nhân 2 nhánh
        self.proj = nn.Conv2d(hidden_ch, out_ch, kernel_size=1, bias=False)
        
        # 3. Nhánh tắt Residual (Nếu in_ch khác out_ch thì dùng Conv 1x1 để khớp size)
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        # Giữ lại identity tensor cho nhánh tắt Residual
        identity = self.shortcut(x)
        
        # Bước 1: Đi qua LayerNorm trước (Pre-LN)
        out = self.ln(x)
        
        # Bước 2: Tính toán cụm SwiGLU song song
        feat = self.w_feat(out)
        gate = self.act(self.w_gate(out))
        
        # Bước 3: Nhân chập 2 nhánh (Gating) và chiếu về out_ch
        swiglu_out = feat * gate
        swiglu_out = self.proj(swiglu_out)
        
        # Bước 4: Cộng kết quả với nhánh tắt ban đầu (Residual Connection)
        return swiglu_out + identity
###

# -------------------------
# wavelet-feature augmentation +  attunet-decoder (lightweight MLP fusion)  
# -------------------------

##
import torch
import torch.nn as nn

class WaveletEdgePerturbation(nn.Module):
    def __init__(self, wave="haar", noise_level=0.2):
        super().__init__()
        self.noise_level = noise_level # Biên độ nhiễu tối đa
        self.dwt = DWTForward(J=1, wave=wave)
        self.idwt = DWTInverse(wave=wave)

    def forward(self, x):
        if not self.training or self.noise_level <= 0:
            return x
            
        Yl, Yh = self.dwt(x)
        Yh_tensor = Yh[0] # Định dạng gốc của bạn [B, C, 3, H, W] hoặc [B, 3, C, H, W]
        
        # Bước 1: Tính toán độ mạnh của biên cạnh (Edge Intensity) từ chính Yh
        # Lấy trị tuyệt đối của Yh, tính trung bình qua các dải LH, HL, HH
        # Kết quả thu được một bản đồ biên cạnh dạng 4D phẳng [B, C, H_half, W_half]
        if Yh_tensor.shape[1] == 3: # Định dạng [B, 3, C, H, W]
            edge_map = torch.abs(Yh_tensor).mean(dim=1)
        else: # Định dạng [B, C, 3, H, W]
            edge_map = torch.abs(Yh_tensor).mean(dim=2)
            
        # Bước 2: Chuẩn hóa edge_map về khoảng [0, 1] để làm Mặt nạ trọng số biên (Edge Mask)
        max_val = edge_map.view(edge_map.size(0), edge_map.size(1), -1).max(dim=-1)[0].view(edge_map.size(0), edge_map.size(1), 1, 1)
        edge_mask = edge_map / (max_val + 1e-6)
        
        # Đẩy thêm dim để khớp broadcast với Yh_tensor gốc của bạn
        if Yh_tensor.shape[1] == 3:
            edge_mask = edge_mask.unsqueeze(1) # [B, 1, C, H, W]
        else:
            edge_mask = edge_mask.unsqueeze(2) # [B, C, 1, H, W]

        # Bước 3: Tạo nhiễu Gauss ngẫu nhiên
        noise = torch.randn_like(Yh_tensor) * self.noise_level
        
        # Bước 4: Nhân nhiễu với edge_mask. 
        # Vùng nào là biên cạnh (giá trị tiến về 1) -> Nhiễu cộng vào rất mạnh.
        # Vùng nào là nước phẳng (giá trị tiến về 0) -> Gần như không có nhiễu.
        Yh_tensor = Yh_tensor + (noise * edge_mask)
        
        out = self.idwt((Yl, [Yh_tensor]))
        return out
###

# -------------------------
# att-unet + high feature guided for feature fusion
# -------------------------



# === Decoder with Cross-Attention and WT-augmentation ===
class AttentionCrossDecoder_WT_ALL(nn.Module):
    def __init__(self, enc_channels, final_channels=64, drop_rate=0.3, separate_channels=True, aug_all=True, aug_feat=False, random_choice= 0.7):
        super().__init__()
        c1, c2, c3, c4 = enc_channels
        self.aug_all = aug_all
        self.aug_feat = aug_feat
        self.random_choice = random_choice
        if self.aug_feat: # do feature level augmentation:  spatial dimension
            self.augmentor = FeatureAugmentor(prob=0.7) 

        # do feature level augmentation:  wavelet dimension
        self.wavelet_mask = WaveletEdgePerturbation(noise_level=drop_rate)
        # Decoder pathway
        self.up4 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        # self.att4 = AttentionGate(F_g=c3, F_l=c3, F_int=c3 // 2)
        self.att4 = CrossAttentionBlock_RoPE(dim_q=c3, dim_kv=c3, num_heads=4, attn_type="global", 
                                             pool_kv=True, pool_stride=2)  # replaced here
        self.conv4 = nn.Sequential(
            PostAttentionSwiGLU(c3 + c3, c3),
            PostAttentionSwiGLU(c3, c3)
        )
        self.up3 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        # self.att3 = AttentionGate(F_g=c2, F_l=c2, F_int=c2 // 2)
        self.att3 = CrossAttentionBlock_RoPE(dim_q=c2, dim_kv=c2, num_heads=4, attn_type="global", 
                                             pool_kv=True, pool_stride=2)  # replaced here
        self.conv3 = nn.Sequential(
            PostAttentionSwiGLU(c2 + c2, c2),
            PostAttentionSwiGLU(c2, c2)
        )

        self.up2 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        # self.att2 = AttentionGate(F_g=c1, F_l=c1, F_int=c1 // 2)
        self.att2 = CrossAttentionBlock_RoPE(dim_q=c1, dim_kv=c1, num_heads=4, attn_type="global", 
                                             pool_kv=True, pool_stride=4)  # replaced here
        self.conv2 = nn.Sequential(
            PostAttentionSwiGLU(c1 + c1, c1),
            PostAttentionSwiGLU(c1, c1)
        )

        self.up1 = nn.ConvTranspose2d(c1, final_channels, kernel_size=2, stride=2)
        # self.att1 = CrossAttentionBlock_RoPE(dim_q=final_channels, dim_kv=final_channels, num_heads=4, attn_type="global", 
        #                                      pool_kv=True, pool_stride=4)
        self.conv1 = nn.Sequential(
            PostAttentionSwiGLU(final_channels, final_channels),
            PostAttentionSwiGLU(final_channels, final_channels)
        )

    def forward(self, feats):
        f1, f2, f3, f4 = feats

        if self.aug_all:
            prob = random.random()
            if prob <= self.random_choice:
                f1 = self.wavelet_mask(f1)
                f2 = self.wavelet_mask(f2)
                f3 = self.wavelet_mask(f3)
                f4 = self.wavelet_mask(f4)

        if self.aug_feat:
            f1, f2, f3, f4 = self.augmentor((f1, f2, f3, f4))

        # --- Stage 4 ---
        d4_up = self.up4(f4)
        if d4_up.shape[-2:] != f3.shape[-2:]:
            d4_up = F.interpolate(d4_up, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        
        f3_att = self.att4(d4_up, f3)
        # Residual Bypass: Cộng f3 nguyên bản với f3_att để bảo toàn chi tiết nhỏ
        # Sau đó concat với d4_up để giữ thông tin toàn cục
        d4 = torch.cat([d4_up, f3 + f3_att], dim=1) 
        d4 = self.conv4(d4)

        # --- Stage 3 ---
        d3_up = self.up3(d4)
        if d3_up.shape[-2:] != f2.shape[-2:]:
            d3_up = F.interpolate(d3_up, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        
        f2_att = self.att3(d3_up, f2)
        d3 = torch.cat([d3_up, f2 + f2_att], dim=1)
        d3 = self.conv3(d3)

        # --- Stage 2 ---
        d2_up = self.up2(d3)
        if d2_up.shape[-2:] != f1.shape[-2:]:
            d2_up = F.interpolate(d2_up, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        
        f1_att = self.att2(d2_up, f1)
        d2 = torch.cat([d2_up, f1 + f1_att], dim=1)
        d2 = self.conv2(d2)

        # --- Stage 1 ---
        d1 = self.up1(d2)
        d1 = self.conv1(d1)
        
        return d1

##



# -------------------------
# Full ConvNeXtUNet_V2 with decoder selector
# -------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DenseCrossMatchingHead(nn.Module):
    """
    Dense pixel-to-pixel cross-attention matching head cho Few-Shot Segmentation.
    Class-agnostic: scorer va cross-attention DUNG CHUNG cho moi class.
    Model hoc "cach so khop", khong "ghi nho class".
    """
    def __init__(self, in_channels, embedding_dim=256, num_heads=4, max_support_per_class=256):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_support_per_class = max_support_per_class

        # Shared projection cho ca query va support -> cung embedding space
        # Dùng GroupNorm thay BatchNorm vì episodic training batch_size=1
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, embedding_dim, 3, padding=1, bias=False),
            nn.GroupNorm(32, embedding_dim),
            nn.ReLU(inplace=True)
        )

        # Cross-attention: query pixels attend den support pixels
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        self.norm_q = nn.LayerNorm(embedding_dim)
        self.norm_s = nn.LayerNorm(embedding_dim)

        # Gated fusion: ket hop query goc + attended features
        self.gate_proj = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid()
        )

        # Shared binary scorer: "pixel nay thuoc class c khong?"
        # DUNG CHUNG cho moi class -> class-agnostic
        self.scorer = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

        # Learnable temperature
        self.logit_scale = nn.Parameter(torch.tensor(15.0))

    def _gather_class_pixels(self, support_proj, support_masks, class_id):
        """Thu thap tat ca projected support pixels thuoc class_id."""
        Ns, D, Hs, Ws = support_proj.shape
        s_flat = support_proj.flatten(2).transpose(1, 2)  # [Ns, HsWs, D]
        mask_flat = (support_masks == class_id).flatten(1)  # [Ns, HsWs]

        pixels_list = []
        for i in range(Ns):
            if mask_flat[i].any():
                pixels_list.append(s_flat[i][mask_flat[i]])

        if not pixels_list:
            return None

        pixels = torch.cat(pixels_list, dim=0)  # [P, D]

        # Subsample neu qua nhieu pixel -> tiet kiem VRAM
        if pixels.size(0) > self.max_support_per_class:
            if self.training:
                idx = torch.randperm(pixels.size(0), device=pixels.device)[:self.max_support_per_class]
            else:
                idx = torch.linspace(0, pixels.size(0) - 1, self.max_support_per_class).long().to(pixels.device)
            pixels = pixels[idx]

        return pixels  # [P, D]

    def forward(self, query_features, support_features, support_masks, targets=None):
        """
        Args:
            query_features:   [B, C_in, H, W] - decoder output cua query
            support_features: [Ns, C_in, Hs, Ws] - decoder output cua support (detached)
            support_masks:    [Ns, Hs, Ws] - class masks tai feature resolution
        Returns:
            logits: [B, num_detected_classes, H, W]
        """
        B, _, H, W = query_features.shape

        # Project ca hai vao cung embedding space (shared weights)
        q_proj = self.proj(query_features)   # [B, D, H, W]
        q_proj = F.normalize(q_proj, p=2, dim=1)
        q_flat = q_proj.flatten(2).transpose(1, 2)  # [B, HW, D]

        s_proj = self.proj(support_features)  # [Ns, D, Hs, Ws]
        s_proj = F.normalize(s_proj, p=2, dim=1)

        # FIX: Tìm index class lớn nhất để đảm bảo số kênh Logits khớp với Targets
        max_s = support_masks[support_masks != 255].max().item() if (support_masks != 255).any() else 0
        if targets is not None:
            max_t = targets[targets != 255].max().item() if (targets != 255).any() else 0
            max_c = max(max_s, max_t)
        else:
            max_c = max_s
            
        num_classes = int(max_c) + 1

        # Thu thập pixel cho tất cả các class
        class_pixels_list = []
        valid_classes = []
        for c in range(num_classes):
            pixels = self._gather_class_pixels(s_proj, support_masks, c)
            if pixels is not None:
                class_pixels_list.append(pixels)
                valid_classes.append(c)

        if len(class_pixels_list) == 0:
            # Không có class nào có support -> trả về toàn logit âm
            return torch.full((B, num_classes, H, W), -10.0, device=q_proj.device)

        # Batching: Tìm max P trong số các class có pixel
        max_p_current = max(p.size(0) for p in class_pixels_list)
        
        num_valid = len(valid_classes)
        D = q_proj.size(1)
        
        s_batched = torch.zeros((num_valid, max_p_current, D), device=q_proj.device)
        pad_mask = torch.zeros((num_valid, max_p_current), dtype=torch.bool, device=q_proj.device)
        
        for i, pixels in enumerate(class_pixels_list):
            P = pixels.size(0)
            s_batched[i, :P] = pixels
            if P < max_p_current:
                pad_mask[i, P:] = True

        # Expand batch dimension cho queries and supports: batch size mới là B * num_valid
        # q_flat: [B, HW, D] -> [B, num_valid, HW, D] -> [B * num_valid, HW, D]
        q_flat_expand = q_flat.unsqueeze(1).expand(B, num_valid, -1, -1).reshape(B * num_valid, -1, D)
        
        # s_batched: [num_valid, max_P, D] -> [B, num_valid, max_P, D] -> [B * num_valid, max_p_current, D]
        s_expand = s_batched.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * num_valid, max_p_current, D)
        
        # pad_mask: [num_valid, max_P] -> [B, num_valid, max_P] -> [B * num_valid, max_p_current]
        pad_mask_expand = pad_mask.unsqueeze(0).expand(B, -1, -1).reshape(B * num_valid, max_p_current)

        # Cross attention (batched across classes)
        attended, _ = self.cross_attn(
            self.norm_q(q_flat_expand),
            self.norm_s(s_expand),
            s_expand,
            key_padding_mask=pad_mask_expand
        )  # [B * num_valid, HW, D]

        # Gated fusion
        combined = torch.cat([q_flat_expand, attended], dim=-1)  # [B * num_valid, HW, 2D]
        gate = self.gate_proj(combined)  # [B * num_valid, HW, D]
        fused = q_flat_expand + gate * attended  # Residual + gated attention

        # Score: "pixel nay thuoc class c khong?" - shared scorer
        score = self.scorer(fused)  # [B * num_valid, HW, 1]
        score = score.view(B, num_valid, H, W)  # [B, num_valid, H, W]

        # Reconstruct full logits
        logits = torch.full((B, num_classes, H, W), -10.0, device=q_proj.device)
        for i, c in enumerate(valid_classes):
            logits[:, c, :, :] = score[:, i, :, :]

        scale = self.logit_scale.clamp(max=100.0)
        return logits * scale


class DINO_AugSeg(nn.Module):
    def __init__(self, encoder, num_classes=1, model_type="tiny", decoder_type="cross_guide_wt_unet", use_wt_aug=True, aug_feat=False,initial_random_choice=0.7):
        """
        encoder: pretrained encoder instance exposing `downsample_layers` and `stages`
        model_type: 'tiny'|'small'|'base'|'large' -> sets enc_channels
        decoder_type: 'attention_unet'|'segformer'|'deeplabv3plus'
        """
        super().__init__()
        self.encoder = encoder
        self.use_wt_aug = use_wt_aug
        self.num_classes = num_classes
        self.embedding_dim = 256
        self.random_choice = initial_random_choice
        # by default freeze encoder weights; user can unfreeze later
        for p in self.encoder.parameters():
            p.requires_grad = False
        # self.encoder.eval()
        if model_type in ("tiny", "small"):
            self.enc_channels = [96, 192, 384, 768]
        elif model_type == "base":
            self.enc_channels = [128, 256, 512, 1024]
        elif model_type == "large":
            self.enc_channels = [192, 384, 768, 1536]
        else:
            raise ValueError("unknown model_type")

        # instantiate decoder
        decoder_type = decoder_type.lower()  
        # CG-Fuse + WT-Aug
        # CG-Fuse: do cross attention between encoder features and decoder features
        # WT-Aug: feature level augmenation on wavelet dimension, only used in training
        if decoder_type == "cross_guide_wt_unet": 
            self.decoder = AttentionCrossDecoder_WT_ALL(self.enc_channels, final_channels=64, aug_all=use_wt_aug, aug_feat=aug_feat,random_choice=self.random_choice)
            decoder_out_channels = 64 
        else:
            raise ValueError("decoder_type must be one of 'attention_unet','segformer','deeplabv3plus'")
        # self.base_prototypes = nn.Parameter(torch.randn(self.num_classes, self.embedding_dim))
        # nn.init.normal_(self.base_prototypes, std=0.02)
        self.matching_head = DenseCrossMatchingHead(
            in_channels=decoder_out_channels + self.enc_channels[0],
            embedding_dim=self.embedding_dim,
            num_heads=4,
            max_support_per_class=256
        )
    def extract_features(self, x):
        feats = []
        out = x
        for i, down in enumerate(self.encoder.downsample_layers):
            out = down(out)
            out = self.encoder.stages[i](out)
            feats.append(out)
        return feats
    def extract_support_features(self, support_images, support_masks):
        """
        Trich xuat raw features tu support set — KHONG nen, KHONG K-means.
        Differentiable thong qua matching head projection.

        Args:
            support_images: [Ns, 3, H, W]
            support_masks:  [Ns, H, W]
        Returns:
            support_features: [Ns, C_in, H', W'] — raw decoded features
            masks_resized:    [Ns, H', W'] — masks tai feature resolution
        """
        feats = self.extract_features(support_images)
        dec_out = self.decoder(feats)

        # Multi-scale fusion (giu nguyen logic encoder-decoder)
        f1 = feats[0]
        if f1.shape[-2:] != dec_out.shape[-2:]:
            f1_resized = F.interpolate(f1, size=dec_out.shape[-2:], mode='bilinear', align_corners=False)
        else:
            f1_resized = f1
        support_features = torch.cat([dec_out, f1_resized], dim=1)  # [Ns, C_in, H', W']

        # Resize masks ve feature resolution
        if support_masks.dim() == 2:
            support_masks = support_masks.unsqueeze(0)
        masks_resized = F.interpolate(
            support_masks.float().unsqueeze(1),
            size=dec_out.shape[-2:],
            mode='nearest'
        ).squeeze(1).long()  # [Ns, H', W']

        return support_features, masks_resized
    def update_random_choice(self, new_prob):
        # 1. Cập nhật thông số quản lý ở lớp vỏ ngoài cùng
        self.random_choice = new_prob
        
        # 2. Bắn thẳng thông số sửa vào AttentionCrossDecoder_WT_ALL
        self.decoder.random_choice = new_prob
        
    def forward(self, x, support_features, support_masks, targets=None):
        """
        Args:
            x:                [B, 3, H, W] — query images
            support_features: [Ns, C_in, H', W'] — tu extract_support_features
            support_masks:    [Ns, H', W'] — tu extract_support_features
            targets:          [B, H, W] — query masks (chi dung luc train)
        """
        # 1. Encoder: get features from dinov3
        feats = []
        out = x
        for i, down in enumerate(self.encoder.downsample_layers):
            out = down(out)
            out = self.encoder.stages[i](out)
            feats.append(out)
        f1, f2, f3, f4 = feats

        # 2. Decoder
        dec_out = self.decoder([f1, f2, f3, f4])

        # 3. Concat query features
        if f1.shape[-2:] != dec_out.shape[-2:]:
            f1_resized = F.interpolate(f1, size=dec_out.shape[-2:], mode='bilinear', align_corners=False)
        else:
            f1_resized = f1
        head_input = torch.cat([dec_out, f1_resized], dim=1)

        # 4. Dense cross-matching: query vs support (pixel-to-pixel)
        logits = self.matching_head(head_input, support_features, support_masks, targets=targets)

        # 5. Upsample ve input resolution
        logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        return logits

if __name__ == '__main__':
    import os

    root_path = "/home/gxu/proj1/lesionSeg"
    # load dinov3 model
    # set model path
    REPO_DIR = os.path.join(root_path, "dino_seg")
    MODEL_TYPE = ["large", ] # "small", "base", "large"
    DECODER_TYPE = ["cross_guide_wt_unet", ] 
    NUM = 0
    model_weight_path = os.path.join(root_path, "dino_seg/checkpoint/dinov3_convnext_"+MODEL_TYPE[NUM]+"_pretrain_lvd.pth")
    # DINOv3 ConvNeXt models pretrained on web images
    dinov3_convnext = torch.hub.load(REPO_DIR, 'dinov3_convnext_'+MODEL_TYPE[NUM], source='local', weights=model_weight_path)
    # print(dinov3_convnext)


    # load pretrained convnext_tiny from your repo (head=Identity)
    encoder = dinov3_convnext
    encoder.head = nn.Identity()   # remove classifier head
    device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img_size = 768

    for dec in DECODER_TYPE:
        print("Testing decoder:", dec)
        # build segmentation model
        model = DINO_AugSeg(encoder, num_classes=2, model_type=MODEL_TYPE[NUM], decoder_type=dec)   # 1 = binary mask
        # print(model)
        model = model.to(device)
        # model.eval()
        # test forward
        batch_img = torch.randn((1, 3, img_size, img_size)).to(device)
        batch_support = torch.randn((1, 3, img_size, img_size)).to(device)
        batch_mask = torch.randint(0, 2, (1, 1, img_size, img_size)).to(device)
        with torch.no_grad():
            s_feat, s_mask_r = model.extract_support_features(batch_support, batch_mask)
            pred_mask = model(batch_img, support_features=s_feat, support_masks=s_mask_r)
        print("Output shape:", pred_mask.shape)
