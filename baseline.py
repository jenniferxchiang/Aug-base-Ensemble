#--------------------------- edited ---------------------------#
#--------------------------- edited ---------------------------#

# python baseline.py --gpu 0 --task MI


from sklearn.metrics import roc_auc_score, average_precision_score
import random
import argparse
from ptbxl_dataset import PTBXLWrapper
from model import *
import copy
import numpy as np
import os
import torch
import torch.nn as nn
import torch.utils.data as utils
import torch.nn.functional as F
import pickle
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt

from torch.backends import cudnn
cudnn.deterministic = True
cudnn.benchmark = False
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser(description='ECG Aug Baseline')

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--savefol', type=str, default='baseline0317')
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--train_samp', type=int, default=1000)
parser.add_argument('--task', type=str, default='MI')


args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

SEED = args.seed
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

dataset_wrapper = PTBXLWrapper(args.batch_size)
train_dataloader, val_dataloader, test_dataloader = dataset_wrapper.get_data_loaders(
    args)


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


def model_saver(epoch, student, opt, path):
    torch.save({
        'epoch': epoch,
        'student_sd': student.state_dict(),
        'optim_sd': opt.state_dict(),
    }, path + f'/checkpoint_epoch{epoch}.pt')


def get_save_path():
    modfol = f"""epoch{args.epochs}-task{args.task}"""
    pth = os.path.join(args.savefol, modfol)
    os.makedirs(pth, exist_ok=True)
    return pth


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


def get_preds(dl, enc):
    y_preds = []
    y_trues = []
    enc.eval()
    for i, (xecg, y) in enumerate(dl):
        y_trues.append(y.detach().numpy())
        xecg = xecg.to(device)
        y_pred = enc.forward(xecg)
        y_preds.append(y_pred.cpu().detach().numpy())

    return (np.concatenate(y_preds, axis=0), np.concatenate(y_trues, axis=0))


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
    except ValueError:
        ld['epoch_loss'] = loss
        ld['auc'] = 0
        ld['auprc'] = 0
    print(ld)
    return ld


def train(train_dl, val_dl, test_dl, warp_aug=None):
    loss_meter = AverageMeter()
    num_outputs = 1
    enc = resnet18(num_outputs=num_outputs).to(device)
    print(train_dl, 'train dl')

    optimizer = torch.optim.Adam(enc.parameters(), args.lr)

    if args.checkpoint is None:
        print("No checkpoint! Training from scratch")
        load_ep = 0

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
    patience = 20  # number of epochs to wait before stopping if validation loss doesn't improve
    num_epochs_no_improvement = 0

    for epoch in range(load_ep, args.epochs):
        for i, (xecg, y) in enumerate(train_dl):
            enc.train()
            xecg = xecg.to(device)
            print("Shape of x:", xecg.shape)
            print("Length of x:", len(xecg))
            y = y.to(device)
            print('run')
            loss = get_loss(enc, xecg, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())
            train_ld['loss'].append(loss.item())

        print("Eval at epoch ", epoch)
        lossdict = evaluate(val_dl, enc)
        val_ld = update_lossdict(val_ld, lossdict)
        # print(
        #     f"Epoch {epoch}, epoch_loss: {val_ld['epoch_loss']}, auc_scores: {val_ld['auc']}")
        cur_val_loss = lossdict['epoch_loss']
        if cur_val_loss < best_val_loss:
            best_val_loss = cur_val_loss
            best_model = copy.deepcopy(enc.state_dict())
            num_epochs_no_improvement = 0
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

    # 1 ==== plot the epoch loss and AUC score after each epoch ====

    plt.plot(val_ld['epoch_loss'], label='Validation Loss')
    plt.plot(train_ld['loss'], label='Training Loss')
    plt.plot(val_ld['auc'], label='AUC Score')
    plt.title('Training / Validation Loss and AUC Score per Epoch' +
              args.task + '- BL')
    plt.xlabel('Epoch')
    plt.xlim(0, int(epoch))
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(get_save_path() + 'one.png')
    # plt.show()

    # ==== end plot part ====

    # 2 ==== plot the epoch loss and AUC score after each epoch ====
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
                  args.task + '- BL')
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
    ax2.set_title('AUC Score per Epoch - ' + args.task + '- BL')
    ax2.legend()
    # ---------------------------------------------------------
    # Adjust the layout of the subplots
    plt.tight_layout()
    plt.savefig(get_save_path() + args.task + '_baseline.png')
    # Show the plot
    plt.show()

    # ==== end plot part ====

    import time
    print(time.time())
    print("Evaluating best model...")
    print(args.task)
    enc.load_state_dict(best_model)
    lossdict = evaluate(test_dl, enc)
    print(time.time())
    test_ld = update_lossdict(test_ld, lossdict)
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

res = train(train_dataloader, val_dataloader, test_dataloader)
