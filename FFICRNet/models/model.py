import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
from einops import rearrange
from models.modules import *


#############################################  主网络：低光光场增强网络  #############################################
class Build_LLLFENet(nn.Module):
    def __init__(self, arg):
        super(Build_LLLFENet, self).__init__()
        ang_res = arg.ang_res
        channels = arg.channels
        stage_num = 4

        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.encoder = EPI_SpatialAngularBlock(channels, ang_res)

        self.cascaded_bottlenecks = nn.ModuleList([CascadedFreqBlock(channels, ang_res) for _ in range(stage_num)])

        self.conv_out1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=1, stride=1, padding=0, bias=False))

        self.conv_out2 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, in_lf):
        # [0,1] shape: (b c u v h w)

        # 1. 浅层特征提取与光照调制
        x_4d = rearrange(in_lf, 'b c u v h w -> (b u v) c h w')   # 空间特征提取
        feat_in_4d = self.conv_in(x_4d)             # [(b u v) c h w]
        feat_in_4d = self.encoder(feat_in_4d)       # [(b u v) c h w]

        # 2.渐进式频段交互循环
        feat_out_4d = feat_in_4d
        for cascade_block in self.cascaded_bottlenecks:
            feat_out_4d = cascade_block(feat_out_4d)      # [(b u v) c h w]

        # 3. 解码与输出
        out_x1 = self.conv_out1(feat_out_4d)
        out_x2 = self.conv_out2(feat_out_4d)
        out_lf = out_x1 * x_4d + out_x2
        out_lf = rearrange(out_lf, '(b u v) c h w -> b c u v h w', b=in_lf.shape[0], u=in_lf.shape[2], v=in_lf.shape[3])
        return out_lf + in_lf


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Low-light light field enhancement -- train mode")
    parser.add_argument("--ang_res", type=int, default=5, help="Angular resolution of light field")
    parser.add_argument("--channels", type=int, default=64, help="Number of channels")
    cfg = parser.parse_args()

    net = Build_LLLFENet(cfg)
    from thop import profile
    input = torch.randn(1, 3, 5, 5, 64, 64)
    out = net(input)
    print(out.shape)
    total = sum([param.nelement() for param in net.parameters()])
    flops, params = profile(net, inputs=(input,))
    print('Number of parameters: %.4fM' % (total / 1e6))
    print('Number of FLOPs: %.4fG' % (flops / 1e9))