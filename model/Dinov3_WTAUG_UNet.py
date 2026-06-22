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

# -------------------------
# helper conv block
# -------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.GroupNorm(16, out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

# -------------------------
# wavelet-feature augmentation
# -------------------------
import torch
import torch.nn as nn

class WaveletEdgePerturbation(nn.Module):
    def __init__(self, wave="haar", noise_level=0.2):
        super().__init__()
        self.noise_level = noise_level 
        self.dwt = DWTForward(J=1, wave=wave)
        self.idwt = DWTInverse(wave=wave)

    def forward(self, x):
        if not self.training or self.noise_level <= 0:
            return x
        Yl, Yh = self.dwt(x)
        Yh_tensor = Yh[0] 
        if Yh_tensor.shape[1] == 3: 
            edge_map = torch.abs(Yh_tensor).mean(dim=1)
        else: 
            edge_map = torch.abs(Yh_tensor).mean(dim=2)
            
        max_val = edge_map.view(edge_map.size(0), edge_map.size(1), -1).max(dim=-1)[0].view(edge_map.size(0), edge_map.size(1), 1, 1)
        edge_mask = edge_map / (max_val + 1e-6)
        
        if Yh_tensor.shape[1] == 3:
            edge_mask = edge_mask.unsqueeze(1) 
        else:
            edge_mask = edge_mask.unsqueeze(2) 

        noise = torch.randn_like(Yh_tensor) * self.noise_level
        Yh_tensor = Yh_tensor + (noise * edge_mask)
        out = self.idwt((Yl, [Yh_tensor]))
        return out

# -------------------------
# Class-Agnostic N-Way K-Shot Components
# -------------------------
class ChannelReducer(nn.Module):
    def __init__(self, enc_channels, reduced_dim=128):
        super().__init__()
        self.reducers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, reduced_dim, kernel_size=1, bias=False),
                nn.GroupNorm(16, reduced_dim) # GroupNorm works well for episodic batch size 1
            ) for c in enc_channels
        ])
    def forward(self, feats):
        return [reducer(feat) for reducer, feat in zip(self.reducers, feats)]

class PushPullMultiHeadMatcher(nn.Module):
    def __init__(self, heads=8):
        super().__init__()
        self.heads = heads

    def forward(self, query_feat, support_feat, support_mask_c):
        """
        query_feat: [B, C, H_q, W_q]
        support_feat: [Ns, C, H_s, W_s]
        support_mask_c: [Ns, H_s, W_s] binary mask for class C
        Returns: Dense Push-Pull Multi-Head correlation tensor [B, 2*heads, H_q, W_q]
        """
        B, C, H_q, W_q = query_feat.shape
        Ns, C, H_s, W_s = support_feat.shape
        
        # Ensure C is divisible by heads
        assert C % self.heads == 0, "Channel dim must be divisible by heads"
        C_per_head = C // self.heads
        
        # Reshape to multi-head: [B, Heads, C_per_head, H_q*W_q]
        q_head = query_feat.reshape(B, self.heads, C_per_head, -1)
        q_norm = F.normalize(q_head, p=2, dim=2)
        
        # Reshape Support: [Ns, Heads, C_per_head, H_s*W_s]
        s_head = support_feat.reshape(Ns, self.heads, C_per_head, -1)
        s_norm = F.normalize(s_head, p=2, dim=2)
        
        mask_flat = support_mask_c.reshape(Ns, -1).float() # [Ns, H_s*W_s]
        mask_flat_all = mask_flat.reshape(-1)
        
        # Flatten support spatially
        # [Heads, Ns * H_s * W_s, C_per_head]
        s_norm_flat = s_norm.permute(1, 0, 3, 2).reshape(self.heads, -1, C_per_head)
        
        valid_fg_idx = torch.where(mask_flat_all > 0)[0]
        valid_bg_idx = torch.where(mask_flat_all == 0)[0]
        
        # Subsample masks to prevent OOM if too large (4000 pixels is rich enough for variance)
        max_pixels = 4000
        if len(valid_fg_idx) > max_pixels:
            rand_idx = torch.randperm(len(valid_fg_idx), device=valid_fg_idx.device)[:max_pixels]
            valid_fg_idx = valid_fg_idx[rand_idx]
            
        if len(valid_bg_idx) > max_pixels:
            rand_idx = torch.randperm(len(valid_bg_idx), device=valid_bg_idx.device)[:max_pixels]
            valid_bg_idx = valid_bg_idx[rand_idx]
        
        sim_fg_heads = []
        sim_bg_heads = []
        
        for h in range(self.heads):
            q_h = q_norm[:, h, :, :].transpose(1, 2) # [B, H_q*W_q, C_per_head]
            s_h = s_norm_flat[h] # [Ns*H_s*W_s, C_per_head]
            
            # Foreground matching
            if len(valid_fg_idx) > 0:
                s_h_fg = s_h[valid_fg_idx] # [P_fg, C_per_head]
                sim_fg_matrix = torch.matmul(q_h, s_h_fg.transpose(0, 1)) # [B, H_q*W_q, P_fg]
                max_fg, _ = torch.max(sim_fg_matrix, dim=-1) # [B, H_q*W_q]
            else:
                max_fg = torch.full((B, H_q * W_q), -1.0, device=query_feat.device)
            sim_fg_heads.append(max_fg.reshape(B, 1, H_q, W_q))
            
            # Background matching
            if len(valid_bg_idx) > 0:
                s_h_bg = s_h[valid_bg_idx] # [P_bg, C_per_head]
                sim_bg_matrix = torch.matmul(q_h, s_h_bg.transpose(0, 1)) # [B, H_q*W_q, P_bg]
                max_bg, _ = torch.max(sim_bg_matrix, dim=-1) # [B, H_q*W_q]
            else:
                max_bg = torch.full((B, H_q * W_q), -1.0, device=query_feat.device)
            sim_bg_heads.append(max_bg.reshape(B, 1, H_q, W_q))
            
        sim_fg = torch.cat(sim_fg_heads, dim=1) # [B, 8, H_q, W_q]
        sim_bg = torch.cat(sim_bg_heads, dim=1) # [B, 8, H_q, W_q]
        
        # Final tensor has 16 channels, maintaining strict semantic order
        sim_map = torch.cat([sim_fg, sim_bg], dim=1) # [B, 16, H_q, W_q]
        return sim_map

class Lightweight_ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 1x1 Conv
        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(16, out_channels),
            nn.ReLU(inplace=True)
        )
        # 3x3 Conv, dilation=6
        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.GroupNorm(16, out_channels),
            nn.ReLU(inplace=True)
        )
        # 3x3 Conv, dilation=12
        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.GroupNorm(16, out_channels),
            nn.ReLU(inplace=True)
        )
        # 3x3 Conv, dilation=18
        self.aspp4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.GroupNorm(16, out_channels),
            nn.ReLU(inplace=True)
        )
        # Global Average Pooling
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(16, out_channels),
            nn.ReLU(inplace=True)
        )
        # Output integration
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.GroupNorm(16, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5)
        )

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_pool(x)
        x5 = F.interpolate(x5, size=x.shape[2:], mode='bilinear', align_corners=False)
        x_out = torch.cat((x1, x2, x3, x4, x5), dim=1)
        return self.out_conv(x_out)

class AttentionCrossDecoder_WT_ALL(nn.Module):
    def __init__(self, enc_channels, reduced_dim=128, drop_rate=0.3, aug_all=True, aug_feat=False, random_choice=0.7):
        super().__init__()
        self.aug_all = aug_all
        self.aug_feat = aug_feat
        self.random_choice = random_choice
        
        self.wavelet_mask = WaveletEdgePerturbation(noise_level=drop_rate)
        
        self.reducer = ChannelReducer(enc_channels, reduced_dim)
        self.matcher = PushPullMultiHeadMatcher(heads=8)
        
        c1, c2, c3, c4 = enc_channels
        K = 16 # 8 Foreground heads + 8 Background heads

        
        # Stage 4
        # Input to ASPP: K correlation channels
        self.aspp = Lightweight_ASPP(K, reduced_dim)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Stage 3
        # Input to Conv3: K correlation channels + up4
        self.conv3 = ConvBNReLU(K + reduced_dim, reduced_dim, kernel_size=3)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Stage 2
        # Input to Conv2: K correlation channels + up3
        self.conv2 = ConvBNReLU(K + reduced_dim, reduced_dim, kernel_size=3)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Stage 1
        # Input to Conv1: K correlation channels + up2
        self.conv1 = ConvBNReLU(K + reduced_dim, reduced_dim, kernel_size=3)
        
        # Final Output logic
        self.final_conv = nn.Conv2d(reduced_dim, 1, kernel_size=1)
        
    def forward(self, q_feats, s_feats_list, s_masks_c):
        """
        q_feats: list of 4 query features
        s_feats_list: list of 4 support features
        s_masks_c: [Ns, H, W] support mask for specific class C
        """
        f1, f2, f3, f4 = q_feats
        
        if self.aug_all and self.training:
            if random.random() <= self.random_choice:
                f1, f2, f3, f4 = [self.wavelet_mask(f) for f in [f1, f2, f3, f4]]
                
        # Reduce channels for both Query and Support
        q_reduced = self.reducer([f1, f2, f3, f4])
        s_reduced = self.reducer(s_feats_list)
        
        # Match at each scale
        sims = []
        for i in range(4):
            # Downsample mask
            mask_i = F.interpolate(s_masks_c.float().unsqueeze(1), size=s_reduced[i].shape[-2:], mode='nearest').squeeze(1)
            sim_i = self.matcher(q_reduced[i], s_reduced[i], mask_i)
            sims.append(sim_i)
            
        # Coarse-to-fine decoding
        # Stage 4 (Pure Correlation Input)
        d4 = sims[3]
        d4 = self.aspp(d4)
        
        # Stage 3 (Pure Correlation Input)
        d4_up = self.up4(d4)
        if d4_up.shape[-2:] != sims[2].shape[-2:]:
            d4_up = F.interpolate(d4_up, size=sims[2].shape[-2:], mode="bilinear", align_corners=False)
        d3 = torch.cat([sims[2], d4_up], dim=1)
        d3 = self.conv3(d3)
        
        # Stage 2 (Pure Correlation Input)
        d3_up = self.up3(d3)
        if d3_up.shape[-2:] != sims[1].shape[-2:]:
            d3_up = F.interpolate(d3_up, size=sims[1].shape[-2:], mode="bilinear", align_corners=False)
        d2 = torch.cat([sims[1], d3_up], dim=1)
        d2 = self.conv2(d2)
        
        # Stage 1 (Pure Correlation Input)
        d2_up = self.up2(d2)
        if d2_up.shape[-2:] != sims[0].shape[-2:]:
            d2_up = F.interpolate(d2_up, size=sims[0].shape[-2:], mode="bilinear", align_corners=False)
        d1 = torch.cat([sims[0], d2_up], dim=1)
        d1 = self.conv1(d1)
        
        logit_c = self.final_conv(d1)
        
        # Temperature scaling to make logits sharper
        return logit_c * 10.0

class DINO_AugSeg(nn.Module):
    def __init__(self, encoder, num_classes=1, model_type="tiny", decoder_type="cross_guide_wt_unet", use_wt_aug=True, aug_feat=False,initial_random_choice=0.7):
        super().__init__()
        self.encoder = encoder
        self.use_wt_aug = use_wt_aug
        self.num_classes = num_classes 
        self.random_choice = initial_random_choice
        
        for p in self.encoder.parameters():
            p.requires_grad = False
            
        if model_type in ("tiny", "small"):
            self.enc_channels = [96, 192, 384, 768]
        elif model_type == "base":
            self.enc_channels = [128, 256, 512, 1024]
        elif model_type == "large":
            self.enc_channels = [192, 384, 768, 1536]
        else:
            raise ValueError("unknown model_type")

        # The new decoder integrates matching and decoding
        self.decoder = AttentionCrossDecoder_WT_ALL(self.enc_channels, reduced_dim=128, aug_all=use_wt_aug, random_choice=self.random_choice)
        
        # Background score bias
        self.bg_bias = nn.Parameter(torch.tensor([0.0]))

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
        In the new paradigm, we just extract backbone features and keep them as a list of 4 multi-scale tensors.
        We return support_masks directly.
        """
        s_feats = self.extract_features(support_images)
        return s_feats, support_masks

    def update_random_choice(self, new_prob):
        self.random_choice = new_prob
        self.decoder.random_choice = new_prob
        
    def forward(self, x, support_features, support_masks, targets=None):
        """
        N-way K-shot Inference
        x: [B, 3, H, W] Query
        support_features: List of 4 tensors [Ns, C, H', W']
        support_masks: [Ns, H_s, W_s] masks
        """
        B, _, H, W = x.shape
        q_feats = self.extract_features(x)
        
        max_s = support_masks[support_masks != 255].max().item() if (support_masks != 255).any() else 0
        if targets is not None:
            max_t = targets[targets != 255].max().item() if (targets != 255).any() else 0
            max_c = max(max_s, max_t)
        else:
            max_c = max_s
            
        num_total_classes = int(max_c) + 1
        
        fg_logits = []
        for c in range(1, num_total_classes):
            mask_c = (support_masks == c).long()
            logit_c = self.decoder(q_feats, support_features, mask_c)
            fg_logits.append(logit_c) 
            
        if len(fg_logits) == 0:
            return torch.full((B, num_total_classes, H, W), -10.0, device=x.device)
            
        fg_logits_tensor = torch.cat(fg_logits, dim=1) # [B, n_way, H_q, W_q]
        fg_logits_tensor = F.interpolate(fg_logits_tensor, size=(H, W), mode='bilinear', align_corners=False)
        
        bg_logit = self.bg_bias.view(1, 1, 1, 1).expand(B, 1, H, W)
        logits = torch.cat([bg_logit, fg_logits_tensor], dim=1)
        
        return logits
