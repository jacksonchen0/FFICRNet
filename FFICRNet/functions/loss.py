import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
from functions import Vgg19
import random


# ssim functions
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return (-1) * ssim_map.mean()
    else:
        return (-1) * ssim_map.mean(1).mean(1).mean(1)


#################################### Loss functions ####################################
class cal_pixel_loss(nn.Module):
    def __init__(self):
        super(cal_pixel_loss, self).__init__()

    def forward(self, x, y):
        eps = 1e-6
        diff = x - y
        error = torch.sqrt(diff * diff + eps)
        loss = torch.sum(error) / torch.numel(error)
        return loss


class cal_ssim_loss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(cal_ssim_loss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return 1 + _ssim(img1, img2, window, self.window_size, channel, self.size_average)


class cal_perc_loss(nn.Module):
    def __init__(self, device):
        super(cal_perc_loss, self).__init__()
        self.l1_loss = torch.nn.L1Loss()
        self.vgg19 = Vgg19.Vgg19(requires_grad=False).to(device)

    def forward(self, infer, gt):
        # [b,c,h,w]
        infer_relu = self.vgg19(infer)
        gt_relu = self.vgg19(gt)
        perc_loss = (((self.l1_loss(infer_relu[0], gt_relu[0]) +
                       self.l1_loss(infer_relu[1], gt_relu[1]) +
                       self.l1_loss(infer_relu[2], gt_relu[2]) +
                       self.l1_loss(infer_relu[3], gt_relu[3]) +
                       self.l1_loss(infer_relu[4], gt_relu[4]))))
        return perc_loss / 5.0


class cal_contrast_Loss(nn.Module):
    def __init__(self, device, mode):
        super(cal_contrast_Loss, self).__init__()
        self.mode = mode
        self.loss = torch.nn.L1Loss()
        self.vgg19 = Vgg19.Vgg19(requires_grad=False).to(device)

    def enhance(self, x1, x2):
        if self.mode == '020':
            random_num = (1.15 - 1.10) * torch.rand(1) + 1.10
        elif self.mode == '050':
            random_num = (1.10 - 1.05) * torch.rand(1) + 1.05
        else:
            random_num = (1.00 - 1.00) * torch.rand(1) + 1.00

        coff_map = x2 / (x1 + 1e-8) * random_num.to(x2.device)
        enh_x1 = x1 * coff_map
        return enh_x1

    def degrade(self, x):
        if self.mode == '020':
            low = 0.80
            high = 0.90
        elif self.mode == '050':
            low = 0.75
            high = 0.85
        else:
            low = 0.70
            high = 0.80

        deg_coff = (high - low) * torch.rand(x.shape) + low
        deg_x = x * deg_coff.to(x.device)
        return deg_x

    def forward(self, anchor, positive, negative):
        # [b,3,h,w]
        negative1 = self.enhance(negative, positive)     # 输入图变亮，会有噪声
        negative2 = self.degrade(positive)               # GT图变暗

        anchor_feats = self.vgg19(anchor)
        positive_feats = self.vgg19(positive)
        negative_feats1 = self.vgg19(negative1)
        negative_feats2 = self.vgg19(negative2)

        positive_loss = (self.loss(anchor_feats[0], positive_feats[0]) +
                         self.loss(anchor_feats[1], positive_feats[1]) +
                         self.loss(anchor_feats[2], positive_feats[2]) +
                         self.loss(anchor_feats[3], positive_feats[3]) +
                         self.loss(anchor_feats[4], positive_feats[4]))

        negative_loss1 = (self.loss(anchor_feats[0], negative_feats1[0]) +
                          self.loss(anchor_feats[1], negative_feats1[1]) +
                          self.loss(anchor_feats[2], negative_feats1[2]) +
                          self.loss(anchor_feats[3], negative_feats1[3]) +
                          self.loss(anchor_feats[4], negative_feats1[4]))

        negative_loss2 = (self.loss(anchor_feats[0], negative_feats2[0]) +
                          self.loss(anchor_feats[1], negative_feats2[1]) +
                          self.loss(anchor_feats[2], negative_feats2[2]) +
                          self.loss(anchor_feats[3], negative_feats2[3]) +
                          self.loss(anchor_feats[4], negative_feats2[4]))

        contrastive_loss = positive_loss / (negative_loss1 + negative_loss2 + 1e-8)
        return contrastive_loss


def get_loss_func(opt):
    losses = {}
    losses['pixel_loss'] = cal_pixel_loss()
    losses['ssim_loss'] = cal_ssim_loss()
    losses['perc_loss'] = cal_perc_loss(opt.device)
    losses['contrast_Loss'] = cal_contrast_Loss(opt.device, opt.ll_level)
    return losses


class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num_losses):
        super(AutomaticWeightedLoss, self).__init__()
        self.params = nn.Parameter(torch.ones(num_losses))

    def forward(self, *losses):
        loss_sum = 0
        for i, loss in enumerate(losses):
            # 使用 exp(-p) 来确保权重始终为正，并增加数值稳定性
            p = torch.clamp(self.params[i], min=-10.0, max=10.0)
            weighted_loss = torch.exp(-p) * loss + p
            # weighted_loss = torch.exp(-self.params[i]) * loss + self.params[i]
            loss_sum += weighted_loss
        return loss_sum


# class AutomaticWeightedLoss(nn.Module):
#     def __init__(self, num_losses):
#         super(AutomaticWeightedLoss, self).__init__()
#         self.params = nn.Parameter(torch.zeros(num_losses))
#
#     def forward(self, *losses):
#         loss_sum = 0
#         for i, loss in enumerate(losses):
#             # 对应的数学公式: L = 0.5 * exp(-s) * loss + 0.5 * s
#             weighted_loss = 0.5 * torch.exp(-self.params[i]) * loss + 0.5 * self.params[i]
#             loss_sum += weighted_loss
#         return loss_sum
