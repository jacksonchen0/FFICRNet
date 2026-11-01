import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from os.path import join
from tqdm import tqdm
import time
from datetime import datetime
from collections import defaultdict
import imageio
import matplotlib.pyplot as plt
from models.model import Build_LLLFENet
from functions.load_dataset import TrainSetLoader, TestSetLoader
from functions.loss import get_loss_func, AutomaticWeightedLoss
from functions.utils import setup_seed, mk_dir, to_2d, lfi2mlia
from validate import run_validate


#############################################################################################################
parser = argparse.ArgumentParser(description="Low-light light field enhancement -- train mode")
parser.add_argument("--device", type=str, default='cuda:0', help="GPU setting")
parser.add_argument("--ll_level", type=str, default="50", help="Low-light level")
parser.add_argument("--model_dir", type=str, default="new_checkpoints/", help="Checkpoints path")
parser.add_argument("--train_path", type=str, default="F:\CJ\KBS\\5x5Dataset\Train\TrainingData_", help="Training data path")
parser.add_argument("--test_path", type=str, default="F:\CJ\KBS\\5x5Dataset\Test\TestData_", help="Test data path")
parser.add_argument("--ang_res", type=int, default=5, help="Angular resolution of light field")
parser.add_argument("--patch_size", type=int, default=64, help="Cropped LF patch size, i.e., spatial resolution")

parser.add_argument("--channels", type=int, default=64, help="Number of channels")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
parser.add_argument("--train_epoch", type=int, default=200, help="Number of epochs to train")
parser.add_argument("--n_steps", type=int, default=50, help="Number of epochs to update learning rate")
parser.add_argument("--decay_value", type=float, default=0.5, help="Learning rate decaying factor")

parser.add_argument("--resume_epoch", type=int, default=0, help="Resume from checkpoint epoch")
parser.add_argument("--num_save", type=int, default=1, help="Number of epochs for saving checkpoint")
parser.add_argument("--num_snapshot", type=int, default=1, help="Number of epochs for saving loss figure")
parser.add_argument("--train_save", type=int, default=0, help="Save the image in training")
parser.add_argument("--val_freq", type=int, default=1, help="Frequency of validation")
parser.add_argument("--val_folder", type=str, default="new_val_results/", help="Folder of validation result")
cfg = parser.parse_args()
print(cfg)

########################################################################################################
SEED = 42
setup_seed(SEED)

# Loss functions
loss_all = get_loss_func(cfg)
# print(loss_all)

####################################################################################################
def train(opt, train_loader, test_loader):

    print('==>training')
    start_time = datetime.now()

    train_results_dir = 'training_results/LF' + opt.ll_level + '/'
    if opt.train_save:
        mk_dir(train_results_dir)

    # model save folder
    opt.model_dir = opt.model_dir + opt.ll_level + '/'
    mk_dir(opt.model_dir)

    ###############################################################################################
    # Build model
    print("Building LLLFENet")
    model_train = Build_LLLFENet(opt).to(opt.device)
    total = sum([param.nelement() for param in model_train.parameters()])
    print(("Params: "+str(round(total/1e6, 4))+'M'))

    awl = AutomaticWeightedLoss(num_losses=3)

    ###############################################################################################
    # Optimizer and loss logger
    optimizer = torch.optim.Adam([
        {'params': model_train.parameters(), 'lr': opt.learning_rate},
        {'params': awl.parameters(), 'lr': opt.learning_rate*0.5}])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.n_steps, gamma=opt.decay_value)
    losslogger = defaultdict(list)

    ###############################################################################################
    # Reload previous parameters
    if opt.resume_epoch:
        resume_path = join(opt.model_dir, 'model_epoch_{}.pth'.format(opt.resume_epoch))
        if os.path.isfile(resume_path):
            print("==>Loading model parameters '{}'".format(resume_path))
            checkpoints = torch.load(resume_path, map_location='cuda:0', weights_only=False)
            model_train.load_state_dict(checkpoints['model'])
            optimizer.load_state_dict(checkpoints['optimizer'])
            scheduler.load_state_dict(checkpoints['scheduler'])
            losslogger = checkpoints['losslogger']
            if 'awl' in checkpoints:
                awl.load_state_dict(checkpoints['awl'])
        else:
            print("==> no model found at 'epoch{}'".format(opt.resume_epoch))

    #############################################################################################
    epoch_state = opt.resume_epoch + 1
    for idx_epoch in range(epoch_state, opt.train_epoch + 1):   # epochs
        # Start training
        model_train.train()
        print('Current epoch learning rate: %e' % (optimizer.state_dict()['param_groups'][0]['lr']))

        loss_epoch = 0.      # Total loss per epoch
        for idx_iter, idx_batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            # [b,c,ah,aw,h,w]
            in_lfi, label_lfi = idx_batch[0].to(opt.device), idx_batch[1].to(opt.device)

            ###############################  Forward inference  ###############################
            train_lfi = model_train(in_lfi)     # [b,c,ah,aw,h,w]

            ##############################  Calculate loss  ##############################
            pixel_loss = loss_all['pixel_loss'](to_2d(train_lfi), to_2d(label_lfi))
            ssim_loss = loss_all['ssim_loss'](to_2d(train_lfi), to_2d(label_lfi))
            contrast_loss = loss_all['contrast_Loss'](to_2d(train_lfi), to_2d(label_lfi), to_2d(in_lfi))
            loss = awl(pixel_loss, ssim_loss, contrast_loss)
            # if(idx_iter%500==0):
            #     print(pixel_loss.item(), ssim_loss.item(), contrast_loss.item(), loss.item())

            # Cumulative loss
            loss_epoch += loss.item()

            ############################  Backward and optimize  ############################
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ##########################  Save training results  ##########################
            if opt.train_save:
                if (idx_iter + 1) % 500 == 0 or idx_iter == len(train_loader)-1:
                    in_name = '{}/epoch{}_iter{}_input.jpg'.format(train_results_dir, idx_epoch, idx_iter + 1)
                    infer_name = '{}/epoch{}_iter{}_infer1.jpg'.format(train_results_dir, idx_epoch, idx_iter + 1)
                    label_name = '{}/epoch{}_iter{}_label.jpg'.format(train_results_dir, idx_epoch, idx_iter + 1)
                    # [c,ah,aw,h,w]
                    save_in = (in_lfi[0, :, :, :, :, :].detach().cpu().numpy().clip(0, 1) * 255.0)
                    save_infer = (train_lfi[0, :, :, :, :, :].detach().cpu().numpy().clip(0, 1) * 255.0)
                    save_label = (label_lfi[0, :, :, :, :, :].detach().cpu().numpy().clip(0, 1) * 255.0)

                    # [c,ah,aw,h,w] --> [h*ah,w*aw,3]
                    save_in = lfi2mlia(save_in)
                    save_infer1 = lfi2mlia(save_infer)
                    save_label = lfi2mlia(save_label)

                    imageio.imwrite(in_name, save_in.astype(np.uint8))
                    imageio.imwrite(infer_name, save_infer1.astype(np.uint8))
                    imageio.imwrite(label_name, save_label.astype(np.uint8))

        scheduler.step()

        ####################################  Print loss  ####################################
        losslogger['epoch'].append(idx_epoch)
        losslogger['loss'].append(loss_epoch/len(train_loader))
        elapsed_time = datetime.now() - start_time
        print('Training==>>Epoch: %d,  loss: %s,  elapsed time: %s'
              % (idx_epoch, loss_epoch/len(train_loader), elapsed_time))

        # write loss
        file_name = 'loss_LF' + opt.ll_level + '.txt'
        file_handle = open(file_name, mode='a')
        file_handle.write('epoch: %d,  loss: %s,  elapsed time: %s\n' % (idx_epoch, loss_epoch/len(train_loader), elapsed_time))
        file_handle.close()

        ###############################  Save model's parameters ###############################
        if idx_epoch % opt.num_save == 0:
            model_save_path = join(opt.model_dir, "model_epoch_{}.pth".format(idx_epoch))
            state = {'epoch': idx_epoch, 'model': model_train.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(), 'losslogger': losslogger,
                     'awl': awl.state_dict()}
            torch.save(state, model_save_path)
            print("checkpoints saved to {}".format(model_save_path))

        # save loss figure
        if idx_epoch % opt.num_snapshot == 0:
            plt.figure()
            plt.title('loss')
            plt.plot(losslogger['epoch'], losslogger['loss'])
            plt.savefig(opt.model_dir + "loss.png")
            plt.close('all')

        ####################################  Validate ####################################
        if idx_epoch % opt.val_freq == 0:
            run_validate(opt, test_loader, idx_epoch)


def main(opt):
    train_set = TrainSetLoader(opt)
    train_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size, shuffle=True, num_workers=0)
    print('Loaded {} training image from {}'.format(len(train_loader), opt.train_path + cfg.ll_level + '_LLE_5x5_Data'))

    test_set = TestSetLoader(opt)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=0)
    print('Loaded {} test image from {}'.format(len(test_loader), opt.test_path + cfg.ll_level + '_LLE_5x5_Data'))

    train(opt, train_loader, test_loader)


##############################################
if __name__ == '__main__':
    main(cfg)
