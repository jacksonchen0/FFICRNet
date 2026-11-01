import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
import time
from datetime import datetime
import imageio
from einops import rearrange
import lpips
from models.model import Build_LLLFENet
from functions.load_dataset import TestSetLoader, TestSetLoader_noGT
from functions.utils import mk_dir, LFdivide, LFintegrate, calculate_metrics, crop_patch, merge_patch

############################################################################################################
parser = argparse.ArgumentParser(description="Low-light light field enhancement  -- test mode")
parser.add_argument("--device", type=str, default='cuda:0', help="GPU setting")
parser.add_argument("--ll_level", type=str, default="100", help="Low-light level")
parser.add_argument("--ang_res", type=int, default=5, help="Angular resolution of light field")
parser.add_argument("--channels", type=int, default=64, help="Number of channels")
parser.add_argument("--model_path", type=str, default="checkpoints/", help="Checkpoints path")
parser.add_argument("--test_path", type=str, default="Dataset/5x5Dataset_64x64/Test/TestData_", help="Test data path")
parser.add_argument("--with_gt", type=str, default=True, help="With GT")
parser.add_argument("--is_save", type=str, default=True, help="Save light field image")
parser.add_argument("--crop", type=int, default=1, help="Crop the LF image into LF patches when out of memory")
parser.add_argument("--patch_size", type=int, default=64, help="Cropped LF patch size, i.e., spatial resolution")
parser.add_argument("--mini_batch", type=int, default=1, help="Mini batch during testing (crop)")
parser.add_argument("--output_path", type=str, default="results/", help="Output path")
cfg = parser.parse_args()
print(cfg)


######################################################################################################
def test(opt, test_loader):
    print('==>testing')
    # Pretrained model path
    pretrained_model_path = opt.model_path + opt.ll_level + '/model_epoch_124.pth'
    if not os.path.exists(pretrained_model_path):
        print('Pretrained model folder is not found ')

    opt.output_path = opt.output_path + opt.ll_level + '/'

    # Load pretrained weight
    checkpoints = torch.load(pretrained_model_path, map_location='cuda:0', weights_only=False)
    ckp_dict = checkpoints['model']

    #################################################################################################
    # Build model
    print("Building LLLFENet")
    model_test = Build_LLLFENet(opt).to(opt.device)
    total = sum([param.nelement() for param in model_test.parameters()])
    print(("Params: " + str(round(total / 1e6, 4)) + 'M'))

    print('loaded model from ' + pretrained_model_path)
    model_test_dict = model_test.state_dict()
    ckp_dict_refine = {k: v for k, v in ckp_dict.items() if k in model_test_dict}
    model_test_dict.update(ckp_dict_refine)
    model_test.load_state_dict(model_test_dict)

    # Output path
    mk_dir(opt.output_path)

    ################################################################################################
    # Test
    ave_psnr_value = []
    ave_ssim_value = []
    ave_lpips_value = []
    lpips_measure = lpips.LPIPS(net='vgg', version='0.1').cuda()

    model_test.eval()
    with torch.no_grad():
        for idx_iter, idx_batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            start_time = datetime.now()

            ####################################  Input data  ####################################
            if (opt.with_gt):
                # [b,c,ah,aw,h,w]
                in_lfi, label_lfi = idx_batch[0], idx_batch[1]
            else:
                # [b,c,ah,aw,h,w]
                in_lfi = idx_batch

            sai_h = in_lfi.shape[-2]
            sai_w = in_lfi.shape[-1]

            ##################################  Forward inference  ##################################
            if (opt.crop == 1):
                # multi_sub_lfs = LFdivide(lf=in_lfi[0], patch_size=opt.patch_size, stride=opt.patch_size//2)
                # n1, n2, u, v, c, h, w = multi_sub_lfs.shape
                # multi_sub_lfs = rearrange(multi_sub_lfs, 'n1 n2 u v c h w -> (n1 n2) c u v h w')

                in_lfi = rearrange(in_lfi[0], 'c u v h w -> (h u) (w v) c')
                spa_length = opt.patch_size
                spa_bound = opt.patch_size // 4
                multi_sub_lfs, lf_row_nums, lf_col_nums = crop_patch(in_lfi, opt.ang_res, spa_length,
                                                                     spa_bound)  # [an*h,aw*w,c,num]
                patch_num = multi_sub_lfs.shape[-1]
                h_patch, w_patch = (spa_length + spa_bound) * opt.ang_res, (
                            spa_length + spa_bound) * opt.ang_res  # MLIA

                test_lfi = []
                # num_infer = (n1 * n2) // opt.mini_batch
                num_infer = patch_num // opt.mini_batch
                for idx_infer in range(num_infer):
                    # in_lf_patch = multi_sub_lfs[idx_infer * opt.mini_batch: (idx_infer + 1) * opt.mini_batch, :, :, :, :, :]
                    # test_lf_patch = model_test(in_lf_patch)
                    # test_lfi.append(test_lf_patch)

                    in_lf_patch = torch.tensor(
                        multi_sub_lfs[:, :, :, idx_infer * opt.mini_batch: (idx_infer + 1) * opt.mini_batch])
                    in_lf_patch = rearrange(in_lf_patch, '(h u) (w v) c 1 -> 1 c u v h w', h=h_patch // opt.ang_res,
                                            u=opt.ang_res, w=w_patch // opt.ang_res, v=opt.ang_res)
                    test_lf_patch = model_test(in_lf_patch.to(opt.device))
                    test_lfi.append(test_lf_patch)

                # if (n1 * n2) % opt.mini_batch:
                #     in_lf_patch = multi_sub_lfs[(idx_infer + 1) * opt.mini_batch:, :, :, :, :, :]
                #     test_lf_patch = model_test(in_lf_patch)
                #     test_lfi.append(test_lf_patch)
                if patch_num % opt.mini_batch:
                    in_lf_patch = torch.tensor(multi_sub_lfs[:, :, :, (idx_infer + 1) * opt.mini_batch:])
                    in_lf_patch = rearrange(in_lf_patch, '(h u) (w v) c 1 -> 1 c u v h w', h=h_patch // opt.ang_res,
                                            u=opt.ang_res, w=h_patch // opt.ang_res, v=opt.ang_res)
                    test_lf_patch = model_test(in_lf_patch.to(opt.device))
                    test_lfi.append(test_lf_patch)

                test_lfi = torch.cat(test_lfi, dim=0)
                # test_lfi = rearrange(test_lfi, '(n1 n2) c u v h w -> n1 n2 u v c h w', n1=n1, n2=n2)
                # test_lfi = LFintegrate(test_lfi, patch_size=opt.patch_size, stride=opt.patch_size//2, sai_h=sai_h, sai_w=sai_w)

                test_lfi = rearrange(test_lfi, 'num c u v h w -> (h u) (w v) c num').cpu()
                test_lfi = merge_patch(test_lfi, lf_row_nums, lf_col_nums, sai_h * opt.ang_res, sai_w * opt.ang_res,
                                       opt.ang_res, spa_length, spa_bound, 3)
                test_lfi = rearrange(test_lfi, '(h u) (w v) c -> 1 c u v h w', h=sai_h, u=opt.ang_res, w=sai_w,
                                     v=opt.ang_res)

            else:
                test_lfi = model_test(in_lfi)
                test_lfi = test_lfi[0]

            elapsed_time = datetime.now() - start_time

            if (opt.with_gt):
                #########################################  Calculate metrics  #########################################
                # [c,ah,aw,h,w]
                test_lfi = test_lfi.squeeze(0)
                label_lfi = label_lfi.squeeze(0)

                psnr_value, ssim_value, lpips_value = calculate_metrics(lpips_measure, torch.tensor(test_lfi).clip(0, 1).to(opt.device), label_lfi.to(opt.device))

                print('Test image.%d,  PSNR: %s,  SSIM: %s,  LPIPS: %s,  Elapsed time: %s'
                      % (idx_iter + 1, psnr_value, ssim_value, lpips_value, elapsed_time))

                ave_psnr_value.append(psnr_value)
                ave_ssim_value.append(ssim_value)
                ave_lpips_value.append(lpips_value)

                file_handle = open(opt.output_path + 'quality_score.txt', mode='a')
                file_handle.write('Img.%d,  PSNR: %s,  SSIM: %s,  LPIPS: %s,\n'
                                  % (idx_iter + 1, psnr_value, ssim_value, lpips_value))
                file_handle.close()

            else:
                test_lfi = test_lfi.squeeze(0)  # [c,ah,aw,h,w]

            ###########################################  Save results  ###########################################
            if (opt.is_save):
                test_lfi = rearrange(test_lfi, 'c u v h w -> u v h w c')
                # test_lfi = (test_lfi.cpu().numpy().clip(0, 1) * 255.0).astype(np.uint8)
                test_lfi = (test_lfi.clip(0, 1) * 255.0).astype(np.uint8)

                save_path = opt.output_path + "Scene_" + str(idx_iter + 1).zfill(2)
                mk_dir(save_path)

                for an_u in range(opt.ang_res):
                    for an_v in range(opt.ang_res):
                        sai_save_path = save_path + '/' + 'View' + '_' + str(an_u + 1).zfill(2) + '_' + str(an_v + 1).zfill(2) + '.png'
                        sai_result = test_lfi[an_u, an_v, :, :, :]
                        imageio.imwrite(sai_save_path, sai_result)

    #############################################  Save metrics  #############################################
    if (opt.with_gt):
        ave_psnr_value = float(np.array(ave_psnr_value).mean())
        ave_ssim_value = float(np.array(ave_ssim_value).mean())
        ave_lpips_value = float(np.array(ave_lpips_value).mean())

        print('Average Metrics,  Ave PSNR: %s,  Ave SSIM: %s, Ave LPIPS: %s'
              % (ave_psnr_value, ave_ssim_value, ave_lpips_value))

        file_handle = open(opt.output_path + 'quality_score.txt', mode='a')
        file_handle.write('Average Metrics,  Ave PSNR: %s,  Ave SSIM: %s,  Ave LPIPS: %s,\n'
                          % (ave_psnr_value, ave_ssim_value, ave_lpips_value))
        file_handle.close()


def main(opt):
    if (opt.with_gt):
        test_set = TestSetLoader(opt)
    else:
        test_set = TestSetLoader_noGT(opt)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=0)
    print('Loaded {} test image from {}.h5'.format(len(test_loader), opt.test_path))

    time1 = datetime.now()
    test(opt, test_loader)
    time2 = datetime.now() - time1
    print('testing end, and taking: ', time2)


##############################################
if __name__ == '__main__':
    main(cfg)
