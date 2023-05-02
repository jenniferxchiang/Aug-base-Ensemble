#--------------------------- edited ---------------------------#
#--------------------------- edited ---------------------------#

# python task.py --gpu 0 --task MI   --a --b
# python task.py --gpu 0 --task STTC --a --b
# python task.py --gpu 0 --task CD   --a --b
# python task.py --gpu 0 --task HYP  --a --b
import time
from torch.backends import cudnn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
import json
import pickle
import torch.nn.functional as F
import torch.utils.data as utils
import torch.nn as nn
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
import random
import argparse
from ptbxl_dataset import PTBXLWrapper
from model import *
import aug_policy
import copy
import numpy as np
import os
from functional import *

from sklearn.metrics import f1_score

# 1_RandTemporalWarp # 2_BaselineWander # 3_GaussianNoise # 4_RandCrop           # 5_RandDisplacement
# 6_MagnitudeScale   # 7_TimeMask       # 8_ChannelMask   # 9_PermuteWaveSegment # 10_ConcatWaveSegment# NoOp()

cudnn.deterministic = True
cudnn.benchmark = False


parser = argparse.ArgumentParser(description='ECG Aug')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--savefol', type=str,
                    default="/home/jennifer/DA4ECG/Ver2_tuningHYP/")
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--train_samp', type=int, default=1000)
parser.add_argument('--task', type=str, default='MI')
parser.add_argument('--a', type=str, default="1")
parser.add_argument('--b', type=str, default="1")

args = parser.parse_args()

method = args.a + "x" + args.b
a = args.a
b = args.b
args.savefol = args.savefol + args.a + "x" + args.b
print("a =", a)
print("b =", b)


start = time.time()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

SEED = args.seed
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
dataset_wrapper = PTBXLWrapper(args.batch_size)
train_dataloader, val_dataloader, test_dataloader = dataset_wrapper.get_data_loaders(
    args)

print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('__Number CUDA Devices:', torch.cuda.device_count())
print('__CUDA Device Name:', torch.cuda.get_device_name(0))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ===========================SNR=====================================
def snr(x_ori, x_aug):
    # Calculate SNR^-1
    rms_ori = np.sqrt(np.mean(np.square(x_ori)))
    rms_diff = np.sqrt(np.mean(np.square(x_ori - x_aug)))
    snr_inv = rms_diff / rms_ori
    return snr_inv

# ==================================================================


def model_saver(epoch, student, opt, path):
    torch.save({
        'epoch': epoch,
        'student_sd': student.state_dict(),
        'optim_sd': opt.state_dict(),
    }, path + f'/checkpoint_epoch{epoch}.pt')


def get_save_path():
    modfol = f"""{args.task} {args.epochs} - {method}"""
    pth = os.path.join(args.savefol, modfol)
    os.makedirs(pth, exist_ok=True)
    return pth

    #  ["NoOp", "BaselineWander", "GaussianNoise", "RandCrop", "RandDisplacement", "MagnitudeScale", "RandTemporalWarp"]


loss_obj = torch.nn.BCEWithLogitsLoss()


def get_loss(enc, x_batch_ecg, y_batch):
    yhat = enc.forward(x_batch_ecg)
    y_batch = y_batch.float()
    loss = loss_obj(yhat.squeeze(), y_batch.squeeze())
    return loss
# Utility function to update lossdict


def update_lossdict(lossdict, update, action='append'):
    for k in update.keys():
        if action == 'append':
            if k in lossdict:
                lossdict[k].append(update[k])
            else:
                lossdict[k] = [update[k]]
        elif action == 'sum':
            if k in lossdict:
                lossdict[k] += update[k]
            else:
                lossdict[k] = update[k]
        else:
            raise NotImplementedError
    return lossdict


def evaluate(dl, enc):
    enc.eval()
    ld = {}
    loss = 0
    loss_obj = torch.nn.BCEWithLogitsLoss()
    y_preds = []
    y_trues = []
    pbar = dl
    with torch.no_grad():
        for i, (xecg, y) in enumerate(pbar):
            y_trues.append(y.detach().numpy())
            xecg = xecg.to(device)
            y = y.to(device)
            y_pred = enc.forward(xecg)
            y_preds.append(y_pred.cpu().detach().numpy())
            l = loss_obj(y_pred.squeeze(), y.squeeze().float())
            loss += l.item()
    loss /= len(dl)
    (y_preds, y_trues) = (np.concatenate(
        y_preds, axis=0), np.concatenate(y_trues, axis=0))
    y_preds = np.squeeze(y_preds)
    y_trues = np.squeeze(y_trues)
    try:
        ld['epoch_loss'] = loss
        ld['auc'] = roc_auc_score(y_trues, y_preds, average=None)
        ld['auprc'] = average_precision_score(y_trues, y_preds, average=None)
        ld['f1_score'] = f1_score(y_trues, y_preds > 0.5, average=None)
        ld['average_f1_score'] = f1_score(
            y_trues, y_preds > 0.5, average='macro')
    except ValueError:
        ld['epoch_loss'] = loss
        ld['auc'] = 0
        ld['auprc'] = 0
        ld['f1_score'] = [0] * y_trues.shape[1]
        ld['average_f1_score'] = 0
    print(ld)
    return ld


def do_aug1(xecg, y, aug1):
    xtask = aug1(xecg, y)
    # print('do aug1')
    return xtask


def do_aug2(xecg, y, aug2):
    xtask = aug2(xecg, y)
    # print('do aug2')
    return xtask


def do_no_operation(xecg, y, no_operation):
    xtask = no_operation(xecg, y)
    # print('do no_operation')
    return xtask


def train(train_dl, val_dl, test_dl, a, b, warp_aug=None):
    loss_meter = AverageMeter()
    aug1 = aug_policy.first_policy(a,
                                   learn_mag=False, learn_prob=False).to(device)
    aug2 = aug_policy.second_policy(b,
                                    learn_mag=False, learn_prob=False).to(device)
    no_operation = aug_policy.third_policy(
        learn_mag=False, learn_prob=False).to(device)
    num_outputs = 1
    enc = resnet18(num_outputs=num_outputs).to(device)
    print('train dl', train_dl)

    optimizer = torch.optim.Adam(enc.parameters(), args.lr)
    if args.checkpoint is None:
        print("No checkpoint! Training from scratch")
        load_ep = 0
    # else:
    #     ckpt = torch.load(args.checkpoint)
    #     student.load_state_dict(ckpt['student_sd'])
    #     optimizer.load_state_dict(ckpt['optim_sd'])
    #     load_ep = ckpt['epoch'] + 1
    #     print("Loaded from ckpt")

    train_ld = {'loss': []}
    val_ld = {}
    test_ld = {}
    print("Checking if run complete")
    savepath = os.path.join(get_save_path(), 'eval_logs.ckpt')
    if os.path.exists(savepath):
        valaucs = torch.load(savepath)['val_ld']['auc']
        if len(valaucs) == args.epochs:
            print(f"Finished this one {savepath}")
            return
    best_val_loss = np.inf
    best_model = copy.deepcopy(enc.state_dict())
    epoch_loss = []
    auc_scores = []
    snr_inv_list = []

    patience = 20  # number of epochs to wait before stopping if validation loss doesn't improve
    num_epochs_no_improvement = 0

    for epoch in range(load_ep, args.epochs):
        count = 0
        # Training code
        for i, (xecg, y) in enumerate(train_dl):
            enc.train()
            xori = xecg.to(device)
            xori = xori.float()
            xecg = xecg.to(device)
            xecg = xecg.float()
            print("Shape of x:", xecg.shape)
            print("Length of x:", len(xecg))
            y = y.to(device)

            # #------------------------------------------------------
            if i == 0 and count < 80:
                # torch.Size([128, 12, 2496])
                # Perform data augmentation on i=0 batches and keep both the original and augmented data
                x_aug = do_aug2(do_aug1(xecg, y, aug1), y, aug2)
                # x_aug = do_aug1(xecg, y, aug1)
                # Calculate SNR^-1 for augmented data and original data
                for j in range(xecg.shape[0]):
                    # get original data
                    x_ori_j = xecg[j].detach().cpu().numpy()
                    # get augmented data
                    x_aug_j = x_aug[j].detach().cpu().numpy()

                    snr_inv_value = snr(x_ori_j, x_aug_j)
                    # print(f"SNR^-1 for sample {j+1}: {snr_inv_value}")
                    snr_inv_list.append(snr_inv_value)
                # Concatenate original and augmented data
                xecg = torch.cat((xecg, x_aug), dim=0)
                # Duplicate labels for augmented data
                y = torch.cat((y, y), dim=0)
                print(i)
                snr_avg = np.mean(snr_inv_list)
                print(f"Average SNR^-1 for batch: {snr_avg:.3f}")
                count += 1
            else:
                xecg = xecg
                print(i)
                print('no operation')
            # #------------------------------------------------------

            loss = get_loss(enc, xecg, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())
            train_ld['loss'].append(loss.item())

        print("Eval at epoch ", epoch)

        # Validation code
        lossdict = evaluate(val_dl, enc)
        val_ld = update_lossdict(val_ld, lossdict)
        cur_val_loss = lossdict['epoch_loss']
        if cur_val_loss < best_val_loss:
            best_val_loss = cur_val_loss
            best_model = copy.deepcopy(enc.state_dict())
            num_epochs_no_improvement = 0  # Reset the counter if validation loss improves
        else:
            num_epochs_no_improvement += 1
            # Increment the counter if validation loss doesn't improve
         # Early stopping check
        if num_epochs_no_improvement >= patience:
            print(
                f'Validation loss has not improved for {patience} epochs. Stopping training early.')
            break

        # append the epoch loss and AUC score to their respective lists
        epoch_loss.append(val_ld['epoch_loss'])
        auc_scores.append(val_ld['auc'])
        tosave = {
            'train_ld': train_ld,
            'val_ld': val_ld,
        }
        torch.save(tosave, os.path.join(get_save_path(), 'eval_logs.ckpt'))
        torch.save(best_model, os.path.join(
            get_save_path(), 'best_model.ckpt'))
        # reset the loss meter for the next epoch
        loss_meter.reset()

    average_snr_inv = sum(snr_inv_list) / len(snr_inv_list)

    print(
        f"=========      Average SNR^-1: {average_snr_inv:.3f}      =========")
    # 1 ==== plot the epoch loss and AUC score after each epoch ====

    plt.plot(val_ld['epoch_loss'], label='Validation Loss')
    plt.plot(train_ld['loss'], label='Training Loss')
    plt.title('Training / Validation Loss per Epoch' + '-' + method)
    plt.xlabel('Epoch')
    plt.xlim(0, int(epoch))
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(get_save_path()
                + 'one.png')
    # plt.show()

    # Create a 1x2 subplot grid
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot epoch loss
    ax1.plot(val_ld['epoch_loss'], label='val loss')
    ax1.plot(train_ld['loss'], label='train loss')
    ax1.set_xlabel('Epoch')
    ax1.set_xlim([0, int(epoch)])
    # Set the y-axis label
    ax1.set_ylabel('Training / Validation Loss')
    # Set the title of the plot
    ax1.set_title('Training / Validation Loss per Epoch - ' +
                  args.task + '-' + method)
    ax1.legend()
    # ---------------------------------------------------------
    # Plot AUC score
    ax2.plot(val_ld['auc'])
    # Set the x-axis label
    ax2.set_xlabel('Epoch')
    # Set the y-axis label
    ax2.set_ylabel('AUC Score')
    ax2.set_ylim([0, 1])
    # Set the title of the plot
    ax2.set_title('AUC Score per Epoch - ' + args.task + '-' + method)
    ax2.legend()
    # ---------------------------------------------------------
    # Adjust the layout of the subplots
    plt.tight_layout()
    plt.savefig(get_save_path() + 'two.png')
    # Show the plot

    # # ==== end plot part ====

    # # ----------------------- SPECTRUM -------------------------

    #------------------- Time domain -------------------#
    fig, axs = plt.subplots(nrows=12, ncols=1, figsize=(10, 20))
    # extract a single batch (e.g. the first batch)
    batch_idx = 0
    xori_b = xori[batch_idx, :, :]
    xaug_b = xecg[batch_idx, :, :]

    for lead_idx in range(12):
        axs[lead_idx].plot(np.arange(len(xori_b[lead_idx])),
                           xori_b[lead_idx].detach().cpu().numpy(), label='Original ECG')
        axs[lead_idx].set_title(f'Lead {lead_idx + 1} -' + method)
        axs[lead_idx].set_xlabel('Time (s)')
        axs[lead_idx].set_xlim(0, 2496)
        axs[lead_idx].set_ylabel('Signal Amplitude')

    for lead_idx in range(12):
        axs[lead_idx].plot(
            np.arange(len(xaug_b[lead_idx])), xaug_b[lead_idx].detach().cpu().numpy(), label='Augmented ECG')
        axs[lead_idx].set_title(f'Lead {lead_idx + 1} -' + method)
        axs[lead_idx].set_xlabel('Time (s)')
        axs[lead_idx].set_xlim(0, 2496)
        axs[lead_idx].set_ylabel('Signal Amplitude')
        axs[0].legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(get_save_path() + '-wf.png')  # Save the figure
    # # plt.show()

    #------------------- Frequency domain -------------------#

    # fig, axs = plt.subplots(nrows=12, ncols=2, figsize=(15, 30))
    # # extract a single batch (e.g. the first batch)
    # batch_idx = 0
    # xori_b = xori[batch_idx, :, :]
    # xaug_b = x_aug[batch_idx, :, :]

    # for lead_idx in range(12):
    #     # plot time-amplitude
    #     axs[lead_idx, 0].plot(np.arange(len(xori_b[lead_idx])),
    #                         xori_b[lead_idx].detach().cpu().numpy(), label='Original ECG')
    #     axs[lead_idx, 0].plot(np.arange(len(xaug_b[lead_idx])),
    #                         xaug_b[lead_idx].detach().cpu().numpy(), label='Augmented ECG')
    #     axs[lead_idx, 0].set_title(f'Lead {lead_idx + 1}')
    #     axs[lead_idx, 0].set_xlabel('Time (s)')
    #     axs[lead_idx, 0].set_xlim(0, 2496)
    #     axs[lead_idx, 0].set_ylabel('Signal Amplitude')
    #     axs[lead_idx, 0].legend(loc='upper right')

    #     # plot frequency-spectrum
    #     fs = 500  # sampling frequency
    #     N = len(xori_b[lead_idx])  # number of samples
    #     freq = np.fft.fftfreq(N, 1/fs)  # frequency vector
    #     Y_ori = np.fft.fft(xori_b[lead_idx].detach().cpu().numpy())  # FFT of the original signal
    #     Y_aug = np.fft.fft(xaug_b[lead_idx].detach().cpu().numpy())  # FFT of the augmented signal
    #     axs[lead_idx, 1].plot(freq[:N//2], np.abs(Y_ori[:N//2]), label='Original ECG')
    #     axs[lead_idx, 1].plot(freq[:N//2], np.abs(Y_aug[:N//2]), label='Augmented ECG')
    #     axs[lead_idx, 1].set_title(f'Lead {lead_idx + 1}')
    #     axs[lead_idx, 1].set_xlabel('Frequency (Hz)')
    #     axs[lead_idx, 1].set_ylabel('Signal Amplitude')
    #     axs[lead_idx, 1].legend(loc='upper right')

    # plt.tight_layout()
    # plt.savefig(get_save_path() + '-wf-fft.png')  # Save the figure

    # ----------------------- SPECTRUM -------------------------

    end = time.time()
    print("EXECUTE TIME: %f " % (end - start))
    t = time.time()
    t1 = time.localtime(t)
    t2 = time.strftime('%Y/%m/%d %H:%M:%S', t1)
    print("current time:", t2)
    print("Evaluating best model...")
    print(args.task)
    enc.load_state_dict(best_model)
    lossdict = evaluate(test_dl, enc)

    f1_scores_list = lossdict['f1_score'].tolist()
    lossdict['f1_score'] = f1_scores_list

    bestresult = json.dumps(lossdict, indent=2)
    fileObject = open(get_save_path() + '--' + '.json', 'w')
    fileObject.write(bestresult)
    fileObject.close()

    print(time.time())
    test_ld = update_lossdict(test_ld, lossdict)
    print(f"Epoch {epoch}")
    print(f"test_epoch_loss: {test_ld['epoch_loss']}")
    print(f"test_auc_scores: {test_ld['auc']}")
    # print(f"test_aucprc: {test_ld['auprc']}")
    print(f"test_aveg_f1   : {test_ld['average_f1_score']}")

    tosave = {
        'train_ld': train_ld,
        'val_ld': val_ld,
        'test_ld': test_ld,
    }
    torch.save(tosave, os.path.join(get_save_path(), 'eval_logs.ckpt'))


print("Checking if run complete")
savepath = os.path.join(get_save_path(), 'eval_logs.ckpt')
if os.path.exists(savepath):
    valaucs = torch.load(savepath)['val_ld']['auc']
    if len(valaucs) == args.epochs:
        print(f"Finished this one {savepath}")

        import sys
        sys.exit(0)
res = train(train_dataloader, val_dataloader, test_dataloader, a, b)
