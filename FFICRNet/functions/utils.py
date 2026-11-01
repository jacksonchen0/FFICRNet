import torch
import torch.nn.functional as functional
import numpy as np
import scipy.io as sio
from scipy.signal import convolve2d
import os
import random
import math
from einops import rearrange
from math import log10, exp


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mk_dir(required_dir):
    if not os.path.exists(required_dir):
        os.makedirs(required_dir)


def to_2d(in_lfi):
    # [b,c,ah,aw,h,w]
    out_lfi = rearrange(in_lfi, 'b c ah aw h w -> (b ah aw) c h w')
    return out_lfi


def lfi2mlia(in_lfi):
    # [3,ah,aw,h,w] to [h*ah,w*aw,c]
    out_mlia = rearrange(in_lfi, 'c ah aw h w -> (h ah) (w aw) c')
    return out_mlia


def calculate_psnr(img1, img2, max_val=1.0):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * log10(max_val / mse)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def calculate_ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=1.0):
    L = val_range
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    padd = window_size//2
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = functional.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = functional.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = functional.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = functional.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = functional.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def calculate_metrics(lpips_measure, infer, label):
    # [c,ah,aw,h,w]
    score1 = 0.
    score2 = 0.
    score3 = 0.
    _, U, V, _, _ = infer.shape

    # To [-1,1]
    infer_norm = infer / 0.5 - 1.
    label_norm = label / 0.5 - 1.

    for u in range(U):
        for v in range(V):
                score1 += calculate_psnr(infer[:, u, v, :, :].unsqueeze(dim=0), label[:, u, v, :, :].unsqueeze(dim=0), max_val=1.0)
                score2 += calculate_ssim(infer[:, u, v, :, :].unsqueeze(dim=0), label[:, u, v, :, :].unsqueeze(dim=0), val_range=1.0).item()
                score3 += lpips_measure.forward(infer_norm[:, u, v, :, :].unsqueeze(dim=0), label_norm[:, u, v, :, :].unsqueeze(dim=0)).item()
    score1 = score1 / (U * V)
    score2 = score2 / (U * V)
    score3 = score3 / (U * V)
    return score1, score2, score3


def ColorAugmentation(lfi, index):
    # [ah,aw,h,w,c]
    if index == 1:
        order = [0, 1, 2]
    elif index == 2:
        order = [0, 2, 1]
    elif index == 3:
        order = [1, 0, 2]
    elif index == 4:
        order = [1, 2, 0]
    elif index == 5:
        order = [2, 1, 0]
    else:
        order = [2, 0, 1]
    aug_lfi = lfi[:, :, :, :, order]
    return aug_lfi


def ImageExtend(Im, bdr):
    [_, _, h, w] = Im.size()
    Im_lr = torch.flip(Im, dims=[-1])
    Im_ud = torch.flip(Im, dims=[-2])
    Im_diag = torch.flip(Im, dims=[-1, -2])

    Im_up = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_mid = torch.cat((Im_lr, Im, Im_lr), dim=-1)
    Im_down = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_Ext = torch.cat((Im_up, Im_mid, Im_down), dim=-2)
    Im_out = Im_Ext[:, :, h - bdr[0]: 2 * h + bdr[1], w - bdr[2]: 2 * w + bdr[3]]
    return Im_out


def LFdivide(lf, patch_size, stride):
    # [c,ah,aw,h,w]
    _, ah, aw, sai_h, sai_w = lf.shape
    data = rearrange(lf, 'c ah aw h w -> (ah aw) c h w')

    bdr = (patch_size - stride) // 2
    numU = (sai_h + bdr * 2 - 1) // stride
    numV = (sai_w + bdr * 2 - 1) // stride
    data_pad = ImageExtend(data, [bdr, bdr + stride - 1, bdr, bdr + stride - 1])
    subLF = functional.unfold(data_pad, kernel_size=patch_size, stride=stride)
    subLF = rearrange(subLF, '(ah aw) (c h w) (n1 n2) -> n1 n2 ah aw c h w',
                      n1=numU, n2=numV, ah=ah, aw=aw, h=patch_size, w=patch_size)
    return subLF


def LFintegrate(subLFs, patch_size, stride, sai_h, sai_w):
    # [n1 n2,ah,aw,c,h,w]
    bdr = (patch_size - stride) // 2
    outLF = subLFs[:, :, :, :, :, bdr:bdr+stride, bdr:bdr+stride]
    outLF = rearrange(outLF, 'n1 n2 u v c h w -> c u v (n1 h) (n2 w)')
    outLF = outLF[:, :, :, 0:sai_h, 0:sai_w]
    return outLF


def crop_patch(in_lf_image, an, spa_length, spa_bound):

    test_h = int(in_lf_image.shape[0])
    test_w = int(in_lf_image.shape[1])

    crop_lengh = spa_length * an      # Height/Width of MLIA
    crop_bound = spa_bound * an

    row_num = test_h // crop_lengh
    col_num = test_w // crop_lengh

    if test_h%crop_lengh==0:
        row_num = row_num-1
    if test_w%crop_lengh==0:
        col_num = col_num-1

    crop_patch_volume = np.zeros((crop_lengh + crop_bound, crop_lengh + crop_bound, 3, 1), dtype=np.float32)  # [H,W,3,N]

    # left top
    for row_cp in range(row_num):
        for col_cp in range(col_num):
            crop_patch = in_lf_image[row_cp * crop_lengh:(row_cp + 1) * crop_lengh + crop_bound, col_cp * crop_lengh:(col_cp + 1) * crop_lengh + crop_bound, :]
            crop_patch = np.expand_dims(crop_patch, axis=-1)
            crop_patch_volume = np.concatenate([crop_patch_volume, crop_patch], axis=-1)

    h_bound_start = test_h - crop_lengh - crop_bound
    w_bound_start = test_w - crop_lengh - crop_bound

    # right
    for row_cp in range(row_num):
        crop_patch = in_lf_image[row_cp * crop_lengh:(row_cp + 1) * crop_lengh + crop_bound, w_bound_start:, :]
        crop_patch = np.expand_dims(crop_patch, axis=-1)
        crop_patch_volume = np.concatenate([crop_patch_volume, crop_patch], axis=-1)

    # bottom
    for col_cp in range(col_num):
        crop_patch = in_lf_image[h_bound_start:, col_cp * crop_lengh:(col_cp + 1) * crop_lengh + crop_bound, :]
        crop_patch = np.expand_dims(crop_patch, axis=-1)
        crop_patch_volume = np.concatenate([crop_patch_volume, crop_patch], axis=-1)

    # right bottom
    crop_patch = in_lf_image[h_bound_start:, w_bound_start:, :]
    crop_patch = np.expand_dims(crop_patch, axis=-1)
    crop_patch_volume = np.concatenate([crop_patch_volume, crop_patch], axis=-1)

    crop_patch_volume = crop_patch_volume[:, :, :, 1:]
    return crop_patch_volume, row_num, col_num


def merge_patch(in_patch_volume_, rnum, cnum, overall_h, overall_w, an, spa_length, spa_bound, chan):

    h_bound = overall_h - spa_length * an * rnum
    w_bound = overall_w - spa_length * an * cnum

    spa_length = spa_length * an
    spa_bound = spa_bound * an
    spa_bound_sub = spa_bound // 2

    # left top
    rec_lf_img = np.zeros((overall_h, overall_w, chan)).astype(np.float32)
    pvx = 0
    for pvi in range(rnum):
        for pvj in range(cnum):
            if (pvi==0 and pvj==0):
                in_tmp_patch = in_patch_volume_[:, :, :, pvx]

            elif (pvi == 0 and pvj > 0):
                in_pre_patch = in_patch_volume_[:, :, :, pvx - 1]
                in_tmp_patch = in_patch_volume_[:, :, :, pvx]
                in_tmp_patch[:, :spa_bound_sub, :] = in_pre_patch[:, spa_length:spa_length + spa_bound_sub, :]

            elif (pvi>0 and pvj==0):
                in_pre_patch = in_patch_volume_[:, :, :, pvx - cnum]
                in_tmp_patch = in_patch_volume_[:, :, :, pvx]
                in_tmp_patch[:spa_bound_sub, :, :] = in_pre_patch[spa_length:spa_length + spa_bound_sub, :, :]

            else:
                in_pre_patch1 = in_patch_volume_[:, :, :, pvx - 1]
                in_pre_patch2 = in_patch_volume_[:, :, :, pvx - cnum]
                in_tmp_patch = in_patch_volume_[:, :, :, pvx]
                in_tmp_patch[:, :spa_bound_sub, :] = in_pre_patch1[:, spa_length:spa_length + spa_bound_sub, :]
                in_tmp_patch[:spa_bound_sub, :, :] = in_pre_patch2[spa_length:spa_length + spa_bound_sub, :, :]

            rec_lf_img[pvi*spa_length:(pvi+1)*spa_length, pvj*spa_length:(pvj+1)*spa_length, :] = in_tmp_patch[:spa_length, :spa_length, :]
            pvx = pvx + 1

    # right
    for pvk in range(rnum):
        if (pvk==0):
            in_tmp_patch = in_patch_volume_[:, :, :, pvx]
        else:
            in_pre_patch = in_patch_volume_[:, :, :, pvx - 1]
            in_tmp_patch = in_patch_volume_[:, :, :, pvx]
            in_tmp_patch[:spa_bound_sub, :, :] = in_pre_patch[spa_length:spa_length + spa_bound_sub, :, :]

        rec_lf_img[pvk*spa_length:(pvk+1)*spa_length, -w_bound:, :] = in_tmp_patch[:spa_length, -w_bound:, :]
        pvx = pvx + 1

    # bottom
    for pvl in range(cnum):
        if (pvl == 0):
            in_tmp_patch = in_patch_volume_[:, :, :, pvx]
        else:
            in_pre_patch = in_patch_volume_[:, :, :, pvx - 1]
            in_tmp_patch = in_patch_volume_[:, :, :, pvx]
            in_tmp_patch[:, :spa_bound_sub, :] = in_pre_patch[:, spa_length:spa_length + spa_bound_sub, :]

        rec_lf_img[-h_bound:, pvl*spa_length:(pvl+1)*spa_length, :] = in_tmp_patch[-h_bound:, :spa_length, :]
        pvx = pvx + 1

    # right bottom
    in_tmp_patch = in_patch_volume_[:, :, :, pvx]
    rec_lf_img[-h_bound:, -w_bound:, :] = in_tmp_patch[-h_bound:, -w_bound:, :]
    return rec_lf_img


