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
# Attention Gate (UNet attention)
# -------------------------
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: gating signal (from decoder), x: skip connection (from encoder)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# -------------------------
# wavelet-feature augmentation +  attunet-decoder (lightweight MLP fusion)  
# -------------------------
class WaveletRandomMask(nn.Module):
    def __init__(self, wave="haar", drop_rate=0.3, separate_channels=False):
        super().__init__()
        self.drop_rate = drop_rate
        self.dwt = DWTForward(J=1, wave=wave)
        self.idwt = DWTInverse(wave=wave)
        self.separate_channels = separate_channels

    def forward(self, x):
        # Only apply in training mode
        if not self.training or self.drop_rate <= 0:
            return x
        
        Yl, Yh = self.dwt(x)
        Yh = Yh[0] # Bx3xCxHxW
        # Random masks
        mask_lp = (torch.rand_like(Yl) > self.drop_rate).float()
        Yl = Yl * mask_lp
        if self.separate_channels:
            # Yh: high-frequency components [B, C, 3, H/2, W/2] (for J=1).
            # Mask each of the 3 subbands separately
            for i in range(3):
                mask_hp = (torch.rand_like(Yh[:, :, i]) > self.drop_rate).float()
                Yh[:, :, i] = Yh[:, :, i] * mask_hp
        else:
            # Mask all high-frequency components together
            mask_hp = (torch.rand_like(Yh) > self.drop_rate).float()
            Yh = Yh * mask_hp

        out = self.idwt((Yl, [Yh]))
        return out

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
# -------------------------
# Cross-attention bottle decoder (lightweight MLP fusion)
# -------------------------
class CrossAttentionBlock(nn.Module):
    def __init__(self, dim_q, dim_kv, num_heads=4, 
                 attn_type="global",  # "global" or "window"
                 window_size=(7, 7),
                 pre_norm=True,
                 use_residual=True,
                 attn_drop=0.1, proj_drop=0.1,
                 use_rel_pos_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.use_residual = use_residual
        self.attn_type = attn_type
        self.window_size = window_size
        self.pre_norm = pre_norm
        self.use_rel_pos_bias = use_rel_pos_bias

        head_dim = dim_q // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim_q, dim_q)
        self.k_proj = nn.Linear(dim_kv, dim_q)
        self.v_proj = nn.Linear(dim_kv, dim_q)

        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(dim_q, dim_q)
        self.proj_drop = nn.Dropout(proj_drop)

        if pre_norm:
            self.norm_q = nn.LayerNorm(dim_q)
            self.norm_kv = nn.LayerNorm(dim_kv)
        else:
            self.norm = nn.LayerNorm(dim_q)

        if use_rel_pos_bias and attn_type == "window":
            self.rel_pos_bias = RelativePositionBias(window_size, num_heads)
        else:
            self.rel_pos_bias = None

    def forward_window(self, q, kv, H, W):
        """ Windowed attention (Swin-style). """
        B, Nq, Cq = q.shape
        window_h, window_w = self.window_size
        assert H % window_h == 0 and W % window_w == 0, "Feature map must be divisible by window size"

        # reshape into windows
        q = q.view(B, H, W, Cq)
        kv = kv.view(B, H, W, Cq)

        # partition windows
        q_windows = q.unfold(1, window_h, window_h).unfold(2, window_w, window_w)
        kv_windows = kv.unfold(1, window_h, window_h).unfold(2, window_w, window_w)

        # reshape to (num_windows*B, Wh*Ww, C)
        q_windows = q_windows.contiguous().view(-1, window_h * window_w, Cq)
        kv_windows = kv_windows.contiguous().view(-1, window_h * window_w, Cq)

        return q_windows, kv_windows

    def forward(self, q, kv):
        B, Cq, Hq, Wq = q.shape
        B, Ck, Hk, Wk = kv.shape

        # Flatten
        q = q.flatten(2).transpose(1, 2)   # (B, Hq*Wq, Cq)
        kv = kv.flatten(2).transpose(1, 2) # (B, Hk*Wk, Ck)

        if self.pre_norm:
            q = self.norm_q(q)
            kv = self.norm_kv(kv)

        Q = self.q_proj(q)
        K = self.k_proj(kv)
        V = self.v_proj(kv)

        if self.attn_type == "window":
            # reshape into windows
            Q, K = self.forward_window(Q, K, Hq, Wq)
            _, V = self.forward_window(Q, V, Hq, Wq)

        # Multi-head split
        Bq = Q.size(0)  # might be B*num_windows
        Q = Q.reshape(Bq, -1, self.num_heads, Cq // self.num_heads).transpose(1, 2)
        K = K.reshape(Bq, -1, self.num_heads, Cq // self.num_heads).transpose(1, 2)
        V = V.reshape(Bq, -1, self.num_heads, Cq // self.num_heads).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale

        # add relative positional bias if available
        if self.rel_pos_bias is not None:
            bias = self.rel_pos_bias()  # (nH, Wh*Ww, Wh*Ww)
            attn = attn + bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ V).transpose(1, 2).reshape(Bq, -1, Cq)
        out = self.out_proj(out)
        out = self.proj_drop(out)

        if self.attn_type == "window":
            # merge windows back
            out = out.view(B, Hq, Wq, Cq).permute(0, 3, 1, 2).contiguous()
        else:
            out = out.transpose(1, 2).reshape(B, Cq, Hq, Wq)

        if self.use_residual:
            if self.pre_norm:
                out = out + q.transpose(1, 2).reshape(B, Cq, Hq, Wq)
            else:
                out = out + q.transpose(1, 2).reshape(B, Cq, Hq, Wq)
                out = self.norm(out.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(B, Cq, Hq, Wq)

        return out


def kmeans_pytorch(x, K, num_iters=5):
    # x: [N, C]
    if x.size(0) == 0:
        return torch.zeros(K, x.size(1), device=x.device)
    if x.size(0) <= K:
        indices = torch.randint(0, x.size(0), (K,), device=x.device)
        return x[indices]
    
    indices = torch.randperm(x.size(0), device=x.device)[:K]
    centers = x[indices]
    
    for _ in range(num_iters):
        dists = torch.cdist(x, centers)
        labels = torch.argmin(dists, dim=1)
        new_centers = []
        for i in range(K):
            mask = labels == i
            if mask.sum() > 0:
                new_centers.append(x[mask].mean(dim=0))
            else:
                new_centers.append(centers[i])
        centers = torch.stack(new_centers)
    return centers

class SupportGuidedAttention(nn.Module):
    def __init__(self, query_dim, proto_dim=256):
        super().__init__()
        self.k_proj = nn.Linear(proto_dim, query_dim, bias=False)
        self.v_proj = nn.Linear(proto_dim, query_dim, bias=False)
        self.q_proj = nn.Conv2d(query_dim, query_dim, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(query_dim, query_dim, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, prototypes):
        if prototypes is None:
            return x
        B, C, H, W = x.shape
        if prototypes.dim() == 3:
            N, K, C_emb = prototypes.shape
            prototypes_flat = prototypes.view(N * K, C_emb)
        else:
            prototypes_flat = prototypes
            
        # Kiểm tra xem prototype nào là hợp lệ (không phải zero vector)
        is_valid_proto = (prototypes_flat.norm(dim=-1) > 0) # [P]
            
        K_feat = self.k_proj(prototypes_flat) 
        V_feat = self.v_proj(prototypes_flat) 
        
        Q_feat = self.q_proj(x).view(B, C, H * W).transpose(1, 2)
        
        Q_feat = F.normalize(Q_feat, p=2, dim=-1)
        # K_feat zero vectors will remain zero vectors after normalize
        K_feat = F.normalize(K_feat, p=2, dim=-1)
        
        attn = torch.einsum('b m c, p c -> b m p', Q_feat, K_feat) * 10.0
        
        # Mask out invalid prototypes
        attn = attn.masked_fill(~is_valid_proto.unsqueeze(0).unsqueeze(0), -float('inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, 0.0) # An toàn nếu tất cả đều là -inf
        
        out = torch.einsum('b m p, p c -> b m c', attn, V_feat)
        out = out.transpose(1, 2).view(B, C, H, W)
        
        out = self.out_proj(out)
        return x + self.gamma * out

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
        self.wavelet_mask = WaveletRandomMask(drop_rate=drop_rate, separate_channels=separate_channels)
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
        
        self.sg_att4 = SupportGuidedAttention(query_dim=c3)
        self.sg_att3 = SupportGuidedAttention(query_dim=c2)
        self.sg_att2 = SupportGuidedAttention(query_dim=c1)
        self.sg_att1 = SupportGuidedAttention(query_dim=final_channels)

    def forward(self, feats, prototypes=None):
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
        d4 = self.sg_att4(d4, prototypes)

        # --- Stage 3 ---
        d3_up = self.up3(d4)
        if d3_up.shape[-2:] != f2.shape[-2:]:
            d3_up = F.interpolate(d3_up, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        
        f2_att = self.att3(d3_up, f2)
        d3 = torch.cat([d3_up, f2 + f2_att], dim=1)
        d3 = self.conv3(d3)
        d3 = self.sg_att3(d3, prototypes)

        # --- Stage 2 ---
        d2_up = self.up2(d3)
        if d2_up.shape[-2:] != f1.shape[-2:]:
            d2_up = F.interpolate(d2_up, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        
        f1_att = self.att2(d2_up, f1)
        d2 = torch.cat([d2_up, f1 + f1_att], dim=1)
        d2 = self.conv2(d2)
        d2 = self.sg_att2(d2, prototypes)

        # --- Stage 1 ---
        d1 = self.up1(d2)
        d1 = self.conv1(d1)
        d1 = self.sg_att1(d1, prototypes)
        
        return d1

##
# =============================================================================
# 2. ROBUST MULTI-SCALE FUSION VỚI ASPP (Hạng nặng, góc nhìn rộng)
# =============================================================================
class ASPPBlock(nn.Module):
    """
    Atrous Spatial Pyramid Pooling: Cung cấp Multi-scale Receptive Field cực mạnh.
    Đổi lấy số lượng tham số lớn để lấy độ chính xác tối đa.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)
        
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = F.interpolate(self.global_avg_pool(x), size=x.shape[2:], mode='bilinear', align_corners=False)
        out = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return self.out_conv(out)

class RobustMultiScaleFusion(nn.Module):
    def __init__(self, enc_channels, embedding_dim=256, drop_rate=0.3, use_wt_aug=True):
        super().__init__()
        c1, c2, c3, c4 = enc_channels
        self.use_wt_aug = use_wt_aug
        
        self.wavelet_enhancer = WaveletEdgePerturbation(noise_level=drop_rate)
        
        self.proj_c4 = PostAttentionSwiGLU(in_ch=c4, out_ch=embedding_dim)
        self.proj_c3 = PostAttentionSwiGLU(in_ch=c3, out_ch=embedding_dim)
        self.proj_c2 = PostAttentionSwiGLU(in_ch=c2, out_ch=embedding_dim)
        self.proj_c1 = PostAttentionSwiGLU(in_ch=c1, out_ch=embedding_dim)
        
        # Áp dụng ASPP trên khối đã nối
        self.aspp = ASPPBlock(embedding_dim * 4, embedding_dim)

    def forward(self, feats):
        f1, f2, f3, f4 = feats
        
        if self.use_wt_aug and self.training:
            f1 = self.wavelet_enhancer(f1)
            f2 = self.wavelet_enhancer(f2)
            f3 = self.wavelet_enhancer(f3)
            f4 = self.wavelet_enhancer(f4)

        _c4 = self.proj_c4(f4)
        _c3 = self.proj_c3(f3)
        _c2 = self.proj_c2(f2)
        _c1 = self.proj_c1(f1)

        size = _c1.size()[2:] 
        _c4 = F.interpolate(_c4, size=size, mode='bilinear', align_corners=False)
        _c3 = F.interpolate(_c3, size=size, mode='bilinear', align_corners=False)
        _c2 = F.interpolate(_c2, size=size, mode='bilinear', align_corners=False)

        out = torch.cat([_c4, _c3, _c2, _c1], dim=1)
        return self.aspp(out)


# =============================================================================
# 3. K-SHOT PROTOTYPE MEMORY BANK & ATTENTION HEAD
# =============================================================================
class MultiShotAttentionHead(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int = 256, init_temperature: float = 15.0):
        """
        Head hạng nặng cho 5-shot đến 15-shot.
        Nó đối chiếu Query với NGÂN HÀNG Prototypes, lấy đặc trưng tốt nhất, 
        sau đó đưa qua mạng tích chập để làm mượt mặt nạ.
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, embedding_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True)
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * init_temperature)
        
        # Mạng học sâu hậu xử lý (Refinement Convolution)
        # Kết hợp bản đồ Cosine (1 channel) và Đặc trưng gốc (embedding_dim)
        self.refinement_conv = nn.Sequential(
            nn.Conv2d(embedding_dim + 1, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, query_features, prototypes, targets=None, margin=0.15):
        """
        query_features: [B, C, H, W]
        prototypes: [N_classes, K_shots, C] (Ma trận bộ nhớ K-Shot)
        """
        # 1. Chuẩn hóa không gian
        q_emb = F.normalize(self.proj(query_features), p=2, dim=1) 
        p_emb = F.normalize(prototypes, p=2, dim=-1)
        
        B, C, H, W = q_emb.shape
        N_classes = p_emb.shape[0]

        # 2. Xử lý K-Shot (K từ 5 đến 15)
        if p_emb.dim() == 3: # Có K-shot: shape [N, K, C]
            # Tính tương đồng Query với TẤT CẢ K-shot: Output [B, N, K, H, W]
            cosine_sim_all = torch.einsum('b c h w, n k c -> b n k h w', q_emb, p_emb)
            
            # Lấy Max qua trục K-shot (Chọn mẫu giống nhất trong 15 mẫu support)
            cosine_sim, _ = torch.max(cosine_sim_all, dim=2) 
        else: # Fallback 1-shot: shape [N, C]
            cosine_sim = torch.einsum('bchw,nc->bnhw', q_emb, p_emb)

        # 3. Margin (Dành cho Train)
        if self.training and targets is not None:
            valid_mask = (targets != 255)
            valid_targets = torch.clamp(targets, min=0, max=N_classes-1)
            one_hot = F.one_hot(valid_targets, num_classes=N_classes).permute(0, 3, 1, 2)
            cosine_sim = torch.where(valid_mask.unsqueeze(1), cosine_sim - (one_hot * margin), cosine_sim)

        # 4. Khuếch đại Nhiệt độ
        cosine_sim = cosine_sim * self.logit_scale
        
        # 5. Lọc Refinement Hạng Nặng (Duyệt qua từng Class)
        logits_list = []
        for n in range(N_classes):
            sim_map = cosine_sim[:, n:n+1, :, :] # [B, 1, H, W]
            # Nối Bản đồ tương đồng với Đặc trưng Query gốc
            concat_feat = torch.cat([q_emb, sim_map], dim=1) # [B, C+1, H, W]
            class_logit = self.refinement_conv(concat_feat) # [B, 1, H, W]
            logits_list.append(class_logit)
            
        final_logits = torch.cat(logits_list, dim=1) # [B, N_classes, H, W]
        
        return final_logits
###
# -------------------------
# Full ConvNeXtUNet_V2 with decoder selector
# -------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class PrototypeSimilarityHead(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int = 256, init_temperature: float = 15.0):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embedding_dim, kernel_size=1, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(init_temperature))

    def forward(self, decoder_features, prototypes, targets=None, margin=0.15):
        # 1. Chiếu và chuẩn hóa L2
        pixel_embeddings = F.normalize(self.proj(decoder_features), p=2, dim=1) 
        
        # Xác định các prototype hợp lệ (không bị rỗng/zero vector do thiếu ảnh Support)
        is_valid_proto = (prototypes.norm(dim=-1) > 0)
        
        prototypes = F.normalize(prototypes, p=2, dim=-1)

        # 2. Tính Cosine Similarity gốc
        if prototypes.dim() == 3: # Multi-prototype: [N, K, C]
            cosine_sim_all = torch.einsum('b c h w, n k c -> b n k h w', pixel_embeddings, prototypes)
            
            # Khử đi ảo giác (False Positives) từ các zero vectors
            valid_mask = is_valid_proto.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # [1, N, K, 1, 1]
            cosine_sim_all = torch.where(valid_mask, cosine_sim_all, torch.full_like(cosine_sim_all, -float('inf')))
            
            cosine_sim, _ = torch.max(cosine_sim_all, dim=2) # [B, N, H, W]
            
            # Nếu 1 class hoàn toàn vắng mặt trong Support, max sẽ là -inf -> logit = -10.0 (rất âm)
            cosine_sim = torch.nan_to_num(cosine_sim, neginf=-10.0)
            
        elif prototypes.dim() == 2:
            cosine_sim = torch.einsum('b c h w, n c -> b n h w', pixel_embeddings, prototypes)
            valid_mask = is_valid_proto.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # [1, N, 1, 1]
            cosine_sim = torch.where(valid_mask, cosine_sim, torch.full_like(cosine_sim, -10.0))
        else:
            cosine_sim = torch.einsum('b c h w, b n c -> b n h w', pixel_embeddings, prototypes)
            valid_mask = is_valid_proto.unsqueeze(-1).unsqueeze(-1) # [B, N, 1, 1]
            cosine_sim = torch.where(valid_mask, cosine_sim, torch.full_like(cosine_sim, -10.0))

        # =======================================================
        # 3. ÁP DỤNG MARGIN (Chỉ chạy lúc Training)
        # =======================================================
        if self.training and targets is not None:
            targets_resized = F.interpolate(
                targets.float().unsqueeze(1), 
                size=cosine_sim.shape[2:], 
                mode='nearest'
            ).squeeze(1).long() 

            valid_mask = (targets_resized != 255)
            
            num_classes = prototypes.size(0) if prototypes.dim() == 3 else prototypes.size(-2)
            valid_targets = torch.clamp(targets_resized, min=0, max=num_classes-1)
            
            one_hot = F.one_hot(valid_targets, num_classes=num_classes).permute(0, 3, 1, 2)
            
            cosine_sim_margin = cosine_sim - (one_hot * margin)
            
            cosine_sim = torch.where(valid_mask.unsqueeze(1), cosine_sim_margin, cosine_sim)
            
        scale = torch.clamp(self.logit_scale, max=4.605).exp()
        # 4. Khuếch đại bằng Nhiệt độ
        logits = cosine_sim * scale
        return logits

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
        self.out_conv = PrototypeSimilarityHead(decoder_out_channels + self.enc_channels[0], embedding_dim=self.embedding_dim)
    def extract_features(self, x):
        feats = []
        out = x
        for i, down in enumerate(self.encoder.downsample_layers):
            out = down(out)
            out = self.encoder.stages[i](out)
            feats.append(out)
        return feats
    def compute_prototype(self, support_image, support_mask, K=5):
        feats = self.extract_features(support_image) 
        # Support decoding without condition
        dec_out = self.decoder(feats, prototypes=None) 
        
        # Multi-scale fusion for head
        f1 = feats[0]
        if f1.shape[-2:] != dec_out.shape[-2:]:
            f1_resized = F.interpolate(f1, size=dec_out.shape[-2:], mode='bilinear', align_corners=False)
        else:
            f1_resized = f1
        head_input = torch.cat([dec_out, f1_resized], dim=1)
        proj_out = self.out_conv.proj(head_input) 
        
        if support_mask.dim() == 3:
            support_mask = support_mask.unsqueeze(1)
            
        mask_resized = F.interpolate(support_mask.float(), size=proj_out.shape[-2:], mode='nearest')
        
        B, C, H, W = proj_out.shape 
        proj_flat = proj_out.view(B, C, -1).transpose(1, 2) # [B, H*W, C]        
        mask_flat = mask_resized.view(B, 1, -1).transpose(1, 2) # [B, H*W, 1]
        
        prototypes_list = []
        
        # Quét từ 0 đến num_classes - 1
        for class_idx in range(self.num_classes):
            c_mask = (mask_flat == class_idx).squeeze(-1) # [B, H*W]
            # Since batch might be > 1 for support, we gather across all batches
            class_pixels = proj_flat[c_mask] # [N_pixels, C]
            
            if class_pixels.size(0) > 0:
                c_proto = kmeans_pytorch(class_pixels, K=K, num_iters=5) # [K, C]
            else:
                c_proto = torch.zeros(K, C, device=proj_flat.device)
                
            prototypes_list.append(c_proto)
            
        dynamic_prototypes = torch.stack(prototypes_list, dim=0) # [num_classes, K, 256]
        dynamic_prototypes = F.normalize(dynamic_prototypes, dim=-1)
        
        return dynamic_prototypes
    def update_random_choice(self, new_prob):
        # 1. Cập nhật thông số quản lý ở lớp vỏ ngoài cùng
        self.random_choice = new_prob
        
        # 2. Bắn thẳng thông số sửa vào AttentionCrossDecoder_WT_ALL
        self.decoder.random_choice = new_prob
        
    def forward(self, x, prototypes, targets=None):
        # ---- encoder forward (same logic as your original) ----
        feats = []
        out = x
        # 1. Encoder: get features from dinov3
        for i, down in enumerate(self.encoder.downsample_layers):
            out = down(out)
            out = self.encoder.stages[i](out)
            feats.append(out)
        # feats: [f1, f2, f3, f4] (shallow -> deep)
        f1, f2, f3, f4 = feats

        # 2. Decoder: do CG-Fuse (training and testing) and WT-Aug(only in training)
        dec_out = self.decoder([f1,f2,f3,f4], prototypes=prototypes)  
        
        # 3. get the final segmentation results
        if f1.shape[-2:] != dec_out.shape[-2:]:
            f1_resized = F.interpolate(f1, size=dec_out.shape[-2:], mode='bilinear', align_corners=False)
        else:
            f1_resized = f1
        head_input = torch.cat([dec_out, f1_resized], dim=1)
        
        logits = self.out_conv(head_input, prototypes, targets=targets)
        # upsample logits to input resolution
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
        batch_img = torch.randn((1, 3, img_size, img_size))
        batch_img = batch_img.to(device)
        with torch.no_grad():
            pred_mask = model(batch_img)   # [1, 1, 768, 768]
        print(pred_mask.shape)
