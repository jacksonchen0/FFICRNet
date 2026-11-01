import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from sympy import false


#############################################  DWT+IDWT  #############################################

class DWT2D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x00, x01 = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
        x10, x11 = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
        ll = (x00 + x01 + x10 + x11) / 2.0
        lh = (x00 + x01 - x10 - x11) / 2.0
        hl = (x00 - x01 + x10 - x11) / 2.0
        hh = (x00 - x01 - x10 + x11) / 2.0
        return ll, lh, hl, hh

class IDWT2D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ll, lh, hl, hh):
        b, c, h, w = ll.shape
        out = torch.zeros((b, c, h * 2, w * 2), dtype=ll.dtype, device=ll.device)
        out[..., 0::2, 0::2] = (ll + lh + hl + hh) / 2.0
        out[..., 0::2, 1::2] = (ll + lh - hl - hh) / 2.0
        out[..., 1::2, 0::2] = (ll - lh + hl - hh) / 2.0
        out[..., 1::2, 1::2] = (ll - lh - hl + hh) / 2.0
        return out


##########################################  EPI Spatial-Angular Geometry Interaction Module  ##########################################
# Shallow Feature Extraction
class EPI_SpatialAngularBlock(nn.Module):
    def __init__(self, dim, ang_res):
        super().__init__()
        self.ang_res = ang_res
        # Horizontal EPI (U, W) extraction , capture horizontal parallax geometry
        self.epi_horiz_conv = nn.Sequential(
            nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, bias=False))

        # Vertical EPI (V, H) extraction , capture vertical parallax geometry
        self.epi_vert_conv = nn.Sequential(
            nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, bias=False))

        self.fuse = nn.Conv2d(in_channels=dim*2, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # x shape: [(b u v),c,h,w]
        b = x.shape[0]//(self.ang_res**2)

        # (Horizontal)
        x_horiz = rearrange(x, '(b u v) c h w -> (b u) c v h w', b=b, u=self.ang_res, v=self.ang_res)
        feat_horiz = self.epi_horiz_conv(x_horiz)
        feat_horiz = rearrange(feat_horiz, '(b u) c v h w -> (b u v) c h w', b=b, u=self.ang_res)

        # (Vertical)
        x_vert = rearrange(x, '(b u v) c h w -> (b v) c u h w', b=b, u=self.ang_res, v=self.ang_res)
        feat_vert = self.epi_vert_conv(x_vert)
        feat_vert = rearrange(feat_vert, '(b v) c u h w -> (b u v) c h w', b=b, v=self.ang_res)

        # Deep Feature Aggregation
        # [(b u v),2c,h,w]
        feat_cat = torch.cat((feat_horiz, feat_vert), dim=1)
        out = self.fuse(feat_cat)
        return out + x


########################################## Progressive High-Low Frequency Interaction Cascade Module  ##########################################
class CascadedFreqBlock(nn.Module):
    def __init__(self, dim, ang_res):
        super().__init__()
        self.dwt = DWT2D()
        self.idwt = IDWT2D()

        # Low-frequency branch: EPI cascaded MDTA
        self.ll_branch = EPI_LLTransformerBranch(dim, ang_res)

        # High-frequency Branch: Spatial -> EPI Cascade Direction-Aware Network
        self.hf_branch = EPI_HFCNNBranch(dim, ang_res)

        # Cross-Frequency Feature Cross-Attention
        self.interaction = FeatureInteraction(dim)

        self.conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        # x shape: [(b u v),c,h,w]
        ll, lh, hl, hh = self.dwt(x)

        # Low-frequency processing
        ll_feat = self.ll_branch(ll)

        # High-frequency processing
        lh_feat, hl_feat, hh_feat = self.hf_branch(lh, hl, hh)
        hf_feat = torch.cat((lh_feat, hl_feat, hh_feat), dim=1)

        # High-Low Frequency Cross-Guided Fusion
        ll_inter, hf_inter = self.interaction(ll_feat, hf_feat)

        ll_out = ll_inter
        lh_out, hl_out, hh_out = torch.chunk(hf_inter, chunks=3, dim=1)

        out = self.idwt(ll_out, lh_out, hl_out, hh_out)
        out = self.conv(out)
        return out + x


##########################################  Low-frequency branch  ##########################################
# Low-frequency backbone network
class EPI_LLTransformerBranch(nn.Module):
    def __init__(self, dim, ang_res):
        super().__init__()
        self.epi_block = EPI_MDTA_Block(dim, ang_res)

    def forward(self, ll_4d):
        out = self.epi_block(ll_4d)
        return out


# EPI Disparity Global Attention Block
class EPI_MDTA_Block(nn.Module):
    def __init__(self, dim, ang_res):
        super().__init__()
        # Spatial MDTA
        self.ang_res = ang_res
        self.norm_sp1 = LayerNorm2d(dim)
        self.attn_sp = MDTA(dim)
        self.norm_sp2 = LayerNorm2d(dim)
        self.ffn_sp = FFNet(dim)

        # Horizontal EPI MDTA ( V , W)
        self.norm_h1 = LayerNorm2d(dim)
        self.attn_h = MDTA(dim)
        self.norm_h2 = LayerNorm2d(dim)
        self.ffn_h = FFNet(dim)

        # Vertical EPI MDTA ( U , H)
        self.norm_v1 = LayerNorm2d(dim)
        self.attn_v = MDTA(dim)
        self.norm_v2 = LayerNorm2d(dim)
        self.ffn_v = FFNet(dim)

        self.fuse_conv = nn.Conv2d(in_channels=dim*2, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        buv, c, h, w = x.shape
        b = buv//(self.ang_res**2)

        # Spatial Transformer
        x_sp = x
        x_sp = x_sp + self.attn_sp(self.norm_sp1(x_sp))
        x_sp = x_sp + self.ffn_sp(self.norm_sp2(x_sp))

        # Horizontal EPI (V, W)
        x_h = rearrange(x_sp, '(b u v) c h w -> (b u h) c v w', b=b, u=self.ang_res, v=self.ang_res)
        x_h = x_h + self.attn_h(self.norm_h1(x_h))
        x_h = x_h + self.ffn_h(self.norm_h2(x_h))
        x_h = rearrange(x_h, '(b u h) c v w -> (b u v) c h w', b=b, u=self.ang_res, h=h)

        # Vertical EPI (U, H)
        x_v = rearrange(x_sp, '(b u v) c h w -> (b v w) c u h', b=b, u=self.ang_res, v=self.ang_res)
        x_v = x_v + self.attn_v(self.norm_v1(x_v))
        x_v = x_v + self.ffn_v(self.norm_v2(x_v))
        x_v = rearrange(x_v, '(b v w) c u h -> (b u v) c h w', b=b, v=self.ang_res, w=w)

        # Fuse EPI features and add global block residuals
        cat_feat = torch.cat((x_h, x_v), dim=1)  # (buv, c, h, w)
        fused_feat = self.fuse_conv(cat_feat)
        return fused_feat + x


#  MDTA Cross-Channel Global Feature Extraction
class LayerNorm2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return rearrange(x, 'b h w c -> b c h w')


class MDTA(nn.Module):
    def __init__(self, dim, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim*3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(in_channels=dim*3, out_channels=dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=False))
        self.project_out = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class FFNet(nn.Module):
    def __init__(self, dim, expansion=2.66):
        super().__init__()
        hidden_dim = int(dim * expansion)
        self.project_in = nn.Conv2d(in_channels=dim, out_channels=hidden_dim*2, kernel_size=1, stride=1, padding=0, bias=False)
        self.dwconv = nn.Conv2d(in_channels=hidden_dim*2, out_channels=hidden_dim*2, kernel_size=3, stride=1, padding=1, groups=hidden_dim*2, bias=False)
        self.project_out = nn.Conv2d(in_channels=hidden_dim, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        out = self.project_out(F.gelu(x1) * x2)
        return out


##########################################  High-frequency branch  ##########################################
# Space -> EPI Cascade Direction-Aware Network
class EPI_HFCNNBranch(nn.Module):
    def __init__(self, dim, ang_res):
        super().__init__()
        self.ang_res = ang_res
        # 1. Spatial domain direction extraction
        self.spa_extract = SpatialDirectionAwareBlock(dim)

        # 2. EPI domain direction extraction
        self.epi_extract = EPIDirectionAwareBlock(dim, ang_res)

        self.lh_conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False)
        self.hl_conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)
        self.hh_conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False)

    def forward(self, lh, hl, hh):
        # Spatial Domain (H, W) High-Frequency Direction Extraction
        spa_lh, spa_hl, spa_hh = self.spa_extract(lh, hl, hh)

        # EPI plane high-frequency direction extraction
        epi_lh, epi_hl, epi_hh = self.epi_extract(spa_lh, spa_hl, spa_hh)

        # Fuse EPI features
        feat_lh = self.lh_conv(epi_lh) + lh
        feat_hl = self.hl_conv(epi_hl) + hl
        feat_hh = self.hh_conv(epi_hh) + hh
        return feat_lh, feat_hl, feat_hh


class SpatialDirectionAwareBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lh_block = HorizontalResBlock(dim)
        self.hl_block = VerticalResBlock(dim)
        self.hh_block = DiagonalResBlock(dim)

    def forward(self, lh, hl, hh):
        feat_lh = self.lh_block(lh)
        feat_hl = self.hl_block(hl)
        feat_hh = self.hh_block(hh)
        return feat_lh, feat_hl, feat_hh


# Basic Residual Direction Extraction Block (1×3, 3×1)
class HorizontalResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False)

    def forward(self, x):
        buffer = self.conv1(x)
        buffer = self.act(buffer)
        buffer = self.conv2(buffer)
        return buffer + x


class VerticalResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)

    def forward(self, x):
        buffer = self.conv1(x)
        buffer = self.act(buffer)
        buffer = self.conv2(buffer)
        return buffer + x


class DiagonalResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        buffer = self.conv1(x)
        buffer = self.act(buffer)
        buffer = self.conv2(buffer)
        return buffer + x


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        buffer = self.conv1(x)
        buffer = self.act(buffer)
        buffer = self.conv2(buffer)
        return buffer + x


# EPI plane direction extraction block
class EPIDirectionAwareBlock(nn.Module):
    def __init__(self, dim, ang_res):
        super().__init__()
        self.ang_res = ang_res
        self.lh_epi_block = HorizontalEPIResBlock(dim, ang_res)
        self.hl_epi_block = VerticalEPIResBlock(dim, ang_res)
        self.hh_epi_block = DiagonalEPIResBlock(dim, ang_res)

    def forward(self, lh, hl, hh):
        feat_lh = self.lh_epi_block(lh)
        feat_hl = self.hl_epi_block(hl)
        feat_hh = self.hh_epi_block(hh)
        return feat_lh, feat_hl, feat_hh


class HorizontalEPIResBlock(nn.Module):
    def __init__(self, dim, ang_res):
        super().__init__()
        self.ang_res = ang_res
        self.hepi_block = ResBlock(dim)

    def forward(self, x):
        buv, _, h, w = x.shape
        b = (x.shape[0] // (self.ang_res**2))

        # Horizontal EPI (V, W)
        x_h = rearrange(x, '(b u v) c h w -> (b u h) c v w', b=b, u=self.ang_res, v=self.ang_res)
        feat_h = self.hepi_block(x_h)
        out_h = rearrange(feat_h, '(b u h) c v w -> (b u v) c h w', b=b, u=self.ang_res, h=h)
        return out_h


class VerticalEPIResBlock(nn.Module):
    def __init__(self, dim, ang_res):
        super().__init__()
        self.ang_res = ang_res
        self.vepi_block = DiagonalResBlock(dim)

    def forward(self, x):
        buv, _, h, w = x.shape
        b = (x.shape[0] // (self.ang_res**2))

        # Vertical EPI (U, H)
        x_v = rearrange(x, '(b u v) c h w -> (b v w) c u h', b=b, u=self.ang_res, v=self.ang_res)
        feat_v = self.vepi_block(x_v)
        out_v = rearrange(feat_v, '(b v w) c u h -> (b u v) c h w', b=b, v=self.ang_res, w=w)
        return out_v


class DiagonalEPIResBlock(nn.Module):
    def __init__(self, dim, ang_res):
        super().__init__()
        self.ang_res = ang_res
        self.hepi_block = ResBlock(dim)
        self.vepi_block = ResBlock(dim)
        self.fuse_epi = nn.Conv2d(in_channels=dim*2, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        buv, _, h, w = x.shape
        b = (x.shape[0] // (self.ang_res**2))

        x_h = rearrange(x, '(b u v) c h w -> (b u h) c v w', b=b, u=self.ang_res, v=self.ang_res)
        feat_h = self.hepi_block(x_h)
        feat_h = rearrange(feat_h, '(b u h) c v w -> (b u v) c h w', b=b, u=self.ang_res, h=h)

        x_v = rearrange(x, '(b u v) c h w -> (b v w) c u h', b=b, u=self.ang_res, v=self.ang_res)
        feat_v = self.vepi_block(x_v)
        feat_v = rearrange(feat_v, '(b v w) c u h -> (b u v) c h w', b=b, v=self.ang_res, w=w)

        feat_fuse_hv = torch.cat((feat_h, feat_v), dim=1)
        out_hv = self.fuse_epi(feat_fuse_hv)
        return out_hv


############################################  Feature Interaction Module  ##########################################
class FeatureInteraction(nn.Module):
    def __init__(self, dim, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        self.temperature1 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        # Low-frequency (LL) projection layers for generating Q, K, V
        self.q_ll = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False))

        self.kv_ll = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim*2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(in_channels=dim*2, out_channels=dim*2, kernel_size=3, stride=1, padding=1, groups=dim, bias=False))

        # High-frequency (HF) projection layers for generating Q, K, V
        self.q_hf = nn.Sequential(
            nn.Conv2d(in_channels=dim*3, out_channels=dim*3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(in_channels=dim*3, out_channels=dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=False))

        self.kv_hf = nn.Sequential(
            nn.Conv2d(in_channels=dim*3, out_channels=dim*3*2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(in_channels=dim*3*2, out_channels=dim*3*2, kernel_size=3, stride=1, padding=1, groups=dim*3*2, bias=False))

        # Attention projection
        self.proj_ll = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj_hf = nn.Conv2d(in_channels=dim*3, out_channels=dim*3, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, ll, hf):
        b, c, h, w = ll.shape

        # HF guides LL
        q_l = self.q_ll(ll)
        q_l = rearrange(q_l, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # HF as Key, Value 
        k_h, v_h = self.kv_hf(hf).chunk(2, dim=1)
        k_h = rearrange(k_h, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_h = rearrange(v_h, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # Attention
        q_l = F.normalize(q_l, dim=-1)
        k_h = F.normalize(k_h, dim=-1)
        attn_ll = (q_l @ k_h.transpose(-2, -1)) * self.temperature1
        attn_ll = attn_ll.softmax(dim=-1)

        # low-frequency features after structural constraint
        out_ll = rearrange(attn_ll @ v_h, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        ll_inter = self.proj_ll(out_ll) + ll   # Cross-domain residual

        # LL guides HF
        q_h = self.q_hf(hf)
        q_h = rearrange(q_h, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # LL as Key, Value 
        k_l, v_l = self.kv_ll(ll).chunk(2, dim=1)
        k_l = rearrange(k_l, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_l = rearrange(v_l, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_h = F.normalize(q_h, dim=-1)
        k_l = F.normalize(k_l, dim=-1)
        attn_hf = (q_h @ k_l.transpose(-2, -1)) * self.temperature2
        attn_hf = attn_hf.softmax(dim=-1)

        # High-frequency features after illumination constraint
        out_hf = rearrange(attn_hf @ v_l, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        hf_inter = self.proj_hf(out_hf) + hf  # Cross-domain residual
        return ll_inter, hf_inter


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim//reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=dim//reduction, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        avg_x = self.avg_pool(x)
        attn = self.mlp(avg_x)
        out = attn * x
        return out



if __name__ == "__main__":
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    dwt = DWT2D()
    idwt = IDWT2D()

    # im = cv2.imread('E:/LLLFE/L3FENet_DWT/1.png').astype(np.float32) / 255.
    # im = np.mean(im, axis=2)
    # print(im.shape)

    # im = torch.from_numpy(im).to('cuda:0')
    # im = im.unsqueeze(0).unsqueeze(0)

    input = torch.rand(25, 3, 64, 64).to('cuda:0')
    c1, c2, c3, c4 = dwt(input)
    rec = idwt(c1, c2, c3, c4)
    value = torch.mean((rec - input) ** 2)
    print('MSE:', value)

    # c1, c2, c3, c4 = dwt(im)
    # rec = idwt(c1, c2, c3, c4)
    # value = torch.mean((rec - im) ** 2)
    # print('MSE:', value)
    #
    # c2 = c2.squeeze().cpu().detach().numpy()
    # c3 = c3.squeeze().cpu().detach().numpy()

    # plt.imshow(c2)
    # plt.show()

    # plt.imshow(c3)
    # plt.show()
