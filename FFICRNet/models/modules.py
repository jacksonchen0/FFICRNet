import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from sympy import false


#############################################  核心子模块1：DWT+IDWT  #############################################
# 无损下/上采样 (DWT / IDWT)
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


##########################################  核心子模块2：EPI空角几何交互模块  ##########################################
# 用于浅层特征提取
class EPI_SpatialAngularBlock(nn.Module):
    def __init__(self, dim, ang_res):
        super().__init__()
        self.ang_res = ang_res
        # 1. 水平 EPI (U, W) 提取，捕捉水平视差几何
        self.epi_horiz_conv = nn.Sequential(
            nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, bias=False))

        # 2. 垂直 EPI (V, H) 提取，捕捉垂直视差几何
        self.epi_vert_conv = nn.Sequential(
            nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, bias=False))

        # 融合这两种几何特征
        self.fuse = nn.Conv2d(in_channels=dim*2, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # x shape: [(b u v),c,h,w]
        b = x.shape[0]//(self.ang_res**2)

        # === 分支 1: 水平 EPI 处理 (Horizontal) ===
        x_horiz = rearrange(x, '(b u v) c h w -> (b u) c v h w', b=b, u=self.ang_res, v=self.ang_res)
        feat_horiz = self.epi_horiz_conv(x_horiz)
        feat_horiz = rearrange(feat_horiz, '(b u) c v h w -> (b u v) c h w', b=b, u=self.ang_res)

        # === 分支 2: 垂直 EPI 处理 (Vertical) ===
        x_vert = rearrange(x, '(b u v) c h w -> (b v) c u h w', b=b, u=self.ang_res, v=self.ang_res)
        feat_vert = self.epi_vert_conv(x_vert)
        feat_vert = rearrange(feat_vert, '(b v) c u h w -> (b u v) c h w', b=b, v=self.ang_res)

        # === 特征深层聚合 ===
        # [(b u v),2c,h,w]
        feat_cat = torch.cat((feat_horiz, feat_vert), dim=1)
        out = self.fuse(feat_cat)
        return out + x


##########################################  核心子模块3：渐进式高低频交互级联模块  ##########################################
class CascadedFreqBlock(nn.Module):
    def __init__(self, dim, ang_res):
        super().__init__()
        self.dwt = DWT2D()
        self.idwt = IDWT2D()

        # 低频分支：EPI 级联 MDTA
        self.ll_branch = EPI_LLTransformerBranch(dim, ang_res)

        # 高频分支：空间 -> EPI 级联方向感知网络
        self.hf_branch = EPI_HFCNNBranch(dim, ang_res)

        # 跨频段特征交叉注意力
        self.interaction = FeatureInteraction(dim)

        self.conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        # x shape: [(b u v),c,h,w]
        ll, lh, hl, hh = self.dwt(x)

        # 低频处理
        ll_feat = self.ll_branch(ll)

        # 高频处理
        lh_feat, hl_feat, hh_feat = self.hf_branch(lh, hl, hh)
        hf_feat = torch.cat((lh_feat, hl_feat, hh_feat), dim=1)

        # CF-CCA 高低频交叉引导融合
        ll_inter, hf_inter = self.interaction(ll_feat, hf_feat)

        ll_out = ll_inter
        lh_out, hl_out, hh_out = torch.chunk(hf_inter, chunks=3, dim=1)

        out = self.idwt(ll_out, lh_out, hl_out, hh_out)
        out = self.conv(out)
        return out + x


##########################################  低频分支  ##########################################
# 低频主干网络
class EPI_LLTransformerBranch(nn.Module):
    def __init__(self, dim, ang_res):
        super().__init__()
        self.epi_block = EPI_MDTA_Block(dim, ang_res)

    def forward(self, ll_4d):
        out = self.epi_block(ll_4d)
        return out


# EPI 视差全局注意力块
class EPI_MDTA_Block(nn.Module):
    def __init__(self, dim, ang_res):
        super().__init__()
        # 1. 空间 MDTA
        self.ang_res = ang_res
        self.norm_sp1 = LayerNorm2d(dim)
        self.attn_sp = MDTA(dim)
        self.norm_sp2 = LayerNorm2d(dim)
        self.ffn_sp = FFNet(dim)

        # 2. 水平 EPI MDTA (提取 V 和 W)
        self.norm_h1 = LayerNorm2d(dim)
        self.attn_h = MDTA(dim)
        self.norm_h2 = LayerNorm2d(dim)
        self.ffn_h = FFNet(dim)

        # 3. 垂直 EPI MDTA (提取 U 和 H)
        self.norm_v1 = LayerNorm2d(dim)
        self.attn_v = MDTA(dim)
        self.norm_v2 = LayerNorm2d(dim)
        self.ffn_v = FFNet(dim)

        self.fuse_conv = nn.Conv2d(in_channels=dim*2, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        buv, c, h, w = x.shape
        b = buv//(self.ang_res**2)

        # 先进行空间Transformer
        x_sp = x
        x_sp = x_sp + self.attn_sp(self.norm_sp1(x_sp))
        x_sp = x_sp + self.ffn_sp(self.norm_sp2(x_sp))

        # 分支 A：水平 EPI (V, W)
        x_h = rearrange(x_sp, '(b u v) c h w -> (b u h) c v w', b=b, u=self.ang_res, v=self.ang_res)
        x_h = x_h + self.attn_h(self.norm_h1(x_h))
        x_h = x_h + self.ffn_h(self.norm_h2(x_h))
        x_h = rearrange(x_h, '(b u h) c v w -> (b u v) c h w', b=b, u=self.ang_res, h=h)

        # 分支 B：垂直 EPI (U, H)
        x_v = rearrange(x_sp, '(b u v) c h w -> (b v w) c u h', b=b, u=self.ang_res, v=self.ang_res)
        x_v = x_v + self.attn_v(self.norm_v1(x_v))
        x_v = x_v + self.ffn_v(self.norm_v2(x_v))
        x_v = rearrange(x_v, '(b v w) c u h -> (b u v) c h w', b=b, v=self.ang_res, w=w)

        # 融合 EPI 特征并加上全局块残差
        cat_feat = torch.cat((x_h, x_v), dim=1)  # (buv, c, h, w)
        fused_feat = self.fuse_conv(cat_feat)
        return fused_feat + x


#  MDTA 跨通道全局特征提取
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


##########################################  高频分支  ##########################################
# 空间 -> EPI 级联方向感知网络
class EPI_HFCNNBranch(nn.Module):
    def __init__(self, dim, ang_res):
        super().__init__()
        self.ang_res = ang_res
        # 1. 空间域方向提取
        self.spa_extract = SpatialDirectionAwareBlock(dim)

        # 2. EPI 域方向提取
        self.epi_extract = EPIDirectionAwareBlock(dim, ang_res)

        self.lh_conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False)
        self.hl_conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)
        self.hh_conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False)

    def forward(self, lh, hl, hh):
        # 空间域 (H, W) 高频方向提取
        spa_lh, spa_hl, spa_hh = self.spa_extract(lh, hl, hh)

        # EPI域 高频方向提取
        epi_lh, epi_hl, epi_hh = self.epi_extract(spa_lh, spa_hl, spa_hh)

        # 融合 EPI 特征
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


# 基础残差方向提取块 (1x3, 3x1 拆分思想)
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


# EPI平面方向提取块
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

        # 水平 EPI (V, W)
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

        # 垂直 EPI (U, H)
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

        # 分支 B：垂直 EPI (U, H)
        x_v = rearrange(x, '(b u v) c h w -> (b v w) c u h', b=b, u=self.ang_res, v=self.ang_res)
        feat_v = self.vepi_block(x_v)
        feat_v = rearrange(feat_v, '(b v w) c u h -> (b u v) c h w', b=b, v=self.ang_res, w=w)

        feat_fuse_hv = torch.cat((feat_h, feat_v), dim=1)
        out_hv = self.fuse_epi(feat_fuse_hv)
        return out_hv


############################################  特征交互模块  ##########################################
class FeatureInteraction(nn.Module):
    def __init__(self, dim, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        self.temperature1 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 低频 (LL) 生成 Q, K, V 的映射层
        self.q_ll = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False))

        self.kv_ll = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim*2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(in_channels=dim*2, out_channels=dim*2, kernel_size=3, stride=1, padding=1, groups=dim, bias=False))

        # 高频 (HF) 生成 Q, K, V 的映射层
        self.q_hf = nn.Sequential(
            nn.Conv2d(in_channels=dim*3, out_channels=dim*3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(in_channels=dim*3, out_channels=dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=False))

        self.kv_hf = nn.Sequential(
            nn.Conv2d(in_channels=dim*3, out_channels=dim*3*2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(in_channels=dim*3*2, out_channels=dim*3*2, kernel_size=3, stride=1, padding=1, groups=dim*3*2, bias=False))

        # 注意力结果投影
        self.proj_ll = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj_hf = nn.Conv2d(in_channels=dim*3, out_channels=dim*3, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, ll, hf):
        b, c, h, w = ll.shape

        # 边缘引导光照 (HF guides LL): 防止提亮产生光晕
        q_l = self.q_ll(ll)
        q_l = rearrange(q_l, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # HF 作为 Key, Value (提供物理结构先验)
        k_h, v_h = self.kv_hf(hf).chunk(2, dim=1)
        k_h = rearrange(k_h, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_h = rearrange(v_h, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # L2 归一化后计算跨频段 Attention
        q_l = F.normalize(q_l, dim=-1)
        k_h = F.normalize(k_h, dim=-1)
        attn_ll = (q_l @ k_h.transpose(-2, -1)) * self.temperature1
        attn_ll = attn_ll.softmax(dim=-1)

        # 得到结构约束后的低频特征
        out_ll = rearrange(attn_ll @ v_h, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        ll_inter = self.proj_ll(out_ll) + ll   # 跨域残差

        # 光照约束边缘 (LL guides HF): 防止放大暗区噪点
        q_h = self.q_hf(hf)
        q_h = rearrange(q_h, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # LL 作为 Key, Value (提供光照/照度环境)
        k_l, v_l = self.kv_ll(ll).chunk(2, dim=1)
        k_l = rearrange(k_l, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_l = rearrange(v_l, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_h = F.normalize(q_h, dim=-1)
        k_l = F.normalize(k_l, dim=-1)
        attn_hf = (q_h @ k_l.transpose(-2, -1)) * self.temperature2
        attn_hf = attn_hf.softmax(dim=-1)

        # 得到光照约束后的高频特征
        out_hf = rearrange(attn_hf @ v_l, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        hf_inter = self.proj_hf(out_hf) + hf  # 跨域残差
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


# 7. 运行测试代码
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
