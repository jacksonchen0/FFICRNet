import torch
import torch.utils.data as data
import h5py
import numpy as np
import random
import os
from functions.utils import ColorAugmentation


################################  Read Training data  ################################
class TrainSetLoader(data.Dataset):
    def __init__(self, cfg):
        super(TrainSetLoader, self).__init__()
        self.dataset_dir = cfg.train_path + cfg.ll_level + '_GaussPerSAI_5x5'
        self.file_list = os.listdir(self.dataset_dir)
        self.item_num = len(self.file_list)

    def __getitem__(self, index):
        # [ah,aw,h,w,3] and in [0,1]
        file_name = self.dataset_dir + '/' + self.file_list[index]

        with h5py.File(file_name, 'r') as hf:
            ll_lfi = np.array(hf.get('data'))
            gt_lfi = np.array(hf.get('label'))   # (c,w,h,aw,ah)

            ll_lfi = np.transpose(ll_lfi, (4, 3, 2, 1, 0))    # (ah,aw,h,w,c)
            gt_lfi = np.transpose(gt_lfi, (4, 3, 2, 1, 0))

            #############################  4D Augmentation  #############################
            # flip
            if np.random.rand(1) > 0.5:
                ll_lfi = np.flip(np.flip(ll_lfi, 0), 2)
                gt_lfi = np.flip(np.flip(gt_lfi, 0), 2)
            if np.random.rand(1) > 0.5:
                ll_lfi = np.flip(np.flip(ll_lfi, 1), 3)
                gt_lfi = np.flip(np.flip(gt_lfi, 1), 3)
            # rotate
            r_ang = np.random.randint(1, 5)
            ll_lfi = np.rot90(ll_lfi, r_ang, (2, 3))
            ll_lfi = np.rot90(ll_lfi, r_ang, (0, 1))
            gt_lfi = np.rot90(gt_lfi, r_ang, (2, 3))
            gt_lfi = np.rot90(gt_lfi, r_ang, (0, 1))
            # color
            # c_ang = np.random.randint(1, 7)
            # ll_lfi = ColorAugmentation(ll_lfi, c_ang)
            # gt_lfi = ColorAugmentation(gt_lfi, c_ang)

            ##########################  Get input and label  ###########################
            # [3,ah,aw,h,w]
            ll_lfi = np.transpose(ll_lfi, [4, 0, 1, 2, 3])
            gt_lfi = np.transpose(gt_lfi, [4, 0, 1, 2, 3])

            ###########################  Convert to tensor  ###########################
            in_data = torch.from_numpy(ll_lfi.copy())
            in_label = torch.from_numpy(gt_lfi.copy())
        return in_data, in_label

    def __len__(self):
        return self.item_num


    ###################################  Read Test data  ###################################
class TestSetLoader(data.Dataset):
    def __init__(self, cfg):
        super(TestSetLoader, self).__init__()
        self.dataset_dir = cfg.test_path + cfg.ll_level + '_GaussPerSAI_5x5'
        self.file_list = os.listdir(self.dataset_dir)
        self.item_num = len(self.file_list)
        print(self.dataset_dir)

    def __getitem__(self, index):
        # [ah,aw,h,w,3] and in [0,1]
        file_name = self.dataset_dir + '/' + self.file_list[index]

        with h5py.File(file_name, 'r') as hf:
            ll_lfi = np.array(hf.get('data'))
            gt_lfi = np.array(hf.get('label'))  # (c,w,h,aw,ah)

            ##########################  Get input and label  ###########################
            # [3,ah,aw,h,w]
            ll_lfi = np.transpose(ll_lfi, [0, 4, 3, 2, 1])
            gt_lfi = np.transpose(gt_lfi, [0, 4, 3, 2, 1])

            ###########################  Convert to tensor  ###########################
            in_data = torch.from_numpy(ll_lfi.copy())
            in_label = torch.from_numpy(gt_lfi.copy())
        return in_data, in_label

    def __len__(self):
        return self.item_num


#################################  Read Test data  #################################
class TestSetLoader_noGT(data.Dataset):
    def __init__(self, cfg):
        super(TestSetLoader_noGT, self).__init__()
        self.dataset_dir = cfg.test_path
        self.file_list = os.listdir(self.dataset_dir)
        self.item_num = len(self.file_list)

    def __getitem__(self, index):
        # [ah,aw,h,w,3] and in [0,1]
        file_name = self.dataset_dir + '/' + self.file_list[index]

        with h5py.File(file_name, 'r') as hf:
            ll_lfi = np.array(hf.get('data'))

            ##########################  Get input and label  ###########################
            # [3,ah,aw,h,w]
            ll_lfi = np.transpose(ll_lfi, [0, 4, 3, 2, 1])

            ###########################  Convert to tensor  ###########################
            in_data = torch.from_numpy(ll_lfi.copy())
        return in_data

    def __len__(self):
        return self.item_num