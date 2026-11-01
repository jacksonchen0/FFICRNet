import argparse
import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
import imageio
from einops import rearrange
import lpips
from models.model import Build_LLLFENet
from functions.utils import mk_dir, LFdivide, LFintegrate, calculate_metrics, crop_patch, merge_patch


################################################################################################
def run_validate(opt, validate_loader, epoch):

    print('==>validate')

    # Pretrained model path
    pretrained_model_path = opt.model_dir + '/model_epoch_' + str(epoch) + '.pth'
    if not os.path.exists(pretrained_model_path):
        print('Pretrained model folder is not found ')

    # Load pretrained weight
    checkpoints = torch.load(pretrained_model_path, map_location='cuda:0', weights_only=False)
    ckp_dict = checkpoints['model']

    ###########################################################################################
    # Build model
    print("Building LLLFENet")
    model_validate = Build_LLLFENet(opt).to(opt.device)

    print('loaded model from ' + pretrained_model_path)
    model_validate_dict = model_validate.state_dict()
    ckp_dict_refine = {k: v for k, v in ckp_dict.items() if k in model_validate_dict}
    model_validate_dict.update(ckp_dict_refine)
    model_validate.load_state_dict(model_validate_dict)

    #######################################################################################
    # Validate
    model_validate.eval()
    with (torch.no_grad()):

        ave_psnr_value = 0.
        ave_ssim_value = 0.
        ave_lpips_value = 0.
        lpips_measure = lpips.LPIPS(net='vgg', version='0.1').cuda()

        for idx_iter, idx_batch in tqdm(enumerate(validate_loader), total=len(validate_loader)):
            ###############################  Input data  ###############################
            # [b,c,ah,aw,h,w]
            in_lfi, label_lfi = idx_batch[0], idx_batch[1]

            sai_h, sai_w  = in_lfi.shape[-2], in_lfi.shape[-1]

            ##################################  Forward inference  ##################################
            # multi_sub_lfs = LFdivide(lf=in_lfi[0], patch_size=opt.patch_size, stride=opt.patch_size//2)
            # n1, n2, u, v, c, h, w = multi_sub_lfs.shape
            # multi_sub_lfs = rearrange(multi_sub_lfs, 'n1 n2 u v c h w -> (n1 n2) c u v h w')

            in_lfi = rearrange(in_lfi[0], 'c u v h w -> (h u) (w v) c')
            spa_length = opt.patch_size
            spa_bound = opt.patch_size // 4
            multi_sub_lfs, lf_row_nums, lf_col_nums = crop_patch(in_lfi, opt.ang_res, spa_length, spa_bound)   # [an*h,aw*w,c,num]
            patch_num = multi_sub_lfs.shape[-1]
            h_patch, w_patch = (spa_length + spa_bound) * opt.ang_res, (spa_length + spa_bound) * opt.ang_res  # MLIA

            val_lfi = []
            mini_batch = 1
            # num_infer = (n1 * n2) // mini_batch
            num_infer = patch_num // mini_batch
            for idx_infer in range(num_infer):
                # in_lf_patch = multi_sub_lfs[idx_infer * mini_batch: (idx_infer + 1) * mini_batch, :, :, :, :, :]
                # val_lf_patch = model_validate(in_lf_patch.to(opt.device))
                # val_lfi.append(val_lf_patch)

                in_lf_patch = torch.tensor(multi_sub_lfs[:, :, :, idx_infer * mini_batch: (idx_infer + 1) * mini_batch])
                in_lf_patch = rearrange(in_lf_patch, '(h u) (w v) c 1 -> 1 c u v h w',  h=h_patch//opt.ang_res, u=opt.ang_res, w=w_patch//opt.ang_res, v=opt.ang_res)
                val_lf_patch = model_validate(in_lf_patch.to(opt.device))
                val_lfi.append(val_lf_patch)

            # if (n1 * n2) % mini_batch:
            #     in_lf_patch = multi_sub_lfs[(idx_infer + 1) * mini_batch:, :, :, :, :, :]
            #     val_lf_patch = model_validate(in_lf_patch.to(opt.device))
            #     val_lfi.append(val_lf_patch)
            if patch_num % mini_batch:
                in_lf_patch = torch.tensor(multi_sub_lfs[:, :, :, (idx_infer + 1) * mini_batch:])
                in_lf_patch = rearrange(in_lf_patch, '(h u) (w v) c 1 -> 1 c u v h w',  h=h_patch//opt.ang_res, u=opt.ang_res, w=h_patch//opt.ang_res, v=opt.ang_res)
                val_lf_patch = model_validate(in_lf_patch.to(opt.device))
                val_lfi.append(val_lf_patch)

            val_lfi = torch.cat(val_lfi, dim=0)
            # val_lfi = rearrange(val_lfi, '(n1 n2) c u v h w -> n1 n2 u v c h w', n1=n1, n2=n2)
            # val_lfi = LFintegrate(val_lfi, patch_size=opt.patch_size, stride=opt.patch_size//2, sai_h=sai_h, sai_w=sai_w)

            val_lfi = rearrange(val_lfi, 'num c u v h w -> (h u) (w v) c num').cpu()
            val_lfi = merge_patch(val_lfi, lf_row_nums, lf_col_nums, sai_h * opt.ang_res,  sai_w * opt.ang_res, opt.ang_res, spa_length, spa_bound, 3)
            val_lfi = rearrange(val_lfi, '(h u) (w v) c -> 1 c u v h w', h=sai_h, u=opt.ang_res, w=sai_w, v=opt.ang_res)

            #########################################  Calculate metrics  #########################################
            val_lfi = val_lfi.squeeze(0)
            label_lfi = label_lfi.squeeze(0)

            psnr_value, ssim_value, lpips_value = calculate_metrics(lpips_measure, torch.tensor(val_lfi).clip(0, 1).to(opt.device), label_lfi.to(opt.device))

            ave_psnr_value += psnr_value / len(validate_loader)
            ave_ssim_value += ssim_value / len(validate_loader)
            ave_lpips_value += lpips_value / len(validate_loader)

            ###########################################  Save results  ###########################################
            val_lfi = rearrange(val_lfi, 'c u v h w -> u v h w c')
            # val_lfi = (val_lfi.cpu().numpy().clip(0, 1) * 255.0).astype(np.uint8)
            val_lfi = (val_lfi.clip(0, 1) * 255.0).astype(np.uint8)

            val_path = opt.val_folder + 'LF' + opt.ll_level + '/epoch_' + str(epoch) + '/'
            save_path = val_path + "Scene_" + str(idx_iter + 1).zfill(2)
            mk_dir(save_path)

            for an_u in range(opt.ang_res):
                for an_v in range(opt.ang_res):
                    sai_save_path = save_path + '/' + 'View' + '_0' + str(an_u + 1) + '_0' + str(an_v + 1) + '.png'
                    sai_result = val_lfi[an_u, an_v, :, :, :]
                    imageio.imwrite(sai_save_path, sai_result)

        ##############################################  Save metrics  ##############################################
        print('Validate end,  Ave PSNR: %s,  Ave SSIM: %s,  Ave LPIPS: %s' % (ave_psnr_value, ave_ssim_value, ave_lpips_value))

        file_handle = open(opt.val_folder + 'LF' + opt.ll_level + '/quality_score.txt', mode='a')
        file_handle.write('Epoch: %s,  Ave PSNR: %s,  Ave SSIM: %s,  Ave LPIPS: %s\n'
                          % (str(epoch), ave_psnr_value, ave_ssim_value, ave_lpips_value))
        file_handle.close()

        ave_score = ave_psnr_value * 0.01 + ave_ssim_value * 0.5 - ave_lpips_value
        print('Validate end,  Ave Metric: %s' % ave_score)

        file_handle = open(opt.val_folder + 'LF' + opt.ll_level + '/quality_score_final.txt', mode='a')
        file_handle.write('Epoch: %s,  Ave Metric: %s\n'
                          % (str(epoch), ave_score))
        file_handle.close()








