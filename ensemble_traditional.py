import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from model import *
from torch.backends import cudnn
from ptbxl_dataset import *
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import  GridSearchCV, cross_val_score, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.metrics import precision_score, recall_score, accuracy_score
from collections import Counter
import itertools



cudnn.deterministic = True
cudnn.benchmark = False

parser = argparse.ArgumentParser(description='ECG ensemble')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--train_samp', type=int, default=1000)
parser.add_argument('--task', type=str, default='MI')
parser.add_argument('--batch_size', type=int, default=128)
args = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(device)

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
        # ld['epoch_loss'] = loss
        ld['auc'] = roc_auc_score(y_trues, y_preds, average=None)
        ld['auprc'] = average_precision_score(y_trues, y_preds, average=None)
        ld['f1_score'] = f1_score(y_trues, y_preds > 0.5, average=None)
        ld['average_f1_score'] = f1_score(
            y_trues, y_preds > 0.5, average='macro')
        ld['precision'] = precision_score(y_trues, y_preds > 0.5, average=None)
        ld['recall'] = recall_score(y_trues, y_preds > 0.5, average=None)
        ld['accuracy'] = accuracy_score(y_trues, y_preds > 0.5)
    except ValueError:
        # ld['epoch_loss'] = loss
        ld['auc'] = 0
        ld['auprc'] = 0
        ld['f1_score'] = [0] * y_trues.shape[1]
        ld['average_f1_score'] = 0
        ld['precision'] = [0] * y_trues.shape[1]
        ld['recall'] = [0] * y_trues.shape[1]
        ld['accuracy'] = 0
    return ld

# --------------- model list ---------------# --------------- model list ---------------# --------------- model list ---------------
model1 = resnet18(num_outputs=1).to(device)
# model1.load_state_dict(torch.load('BEST/MI_best_model.ckpt'))
model2 = resnet18(num_outputs=1).to(device)
# model2.load_state_dict(torch.load('BEST/STTC_best_model.ckpt'))
model3 = resnet18(num_outputs=1).to(device)
# model3.load_state_dict(torch.load('BEST/CD_best_model.ckpt'))
model4 = resnet18(num_outputs=1).to(device)
# model4.load_state_dict(torch.load('BEST/HYP_best_model.ckpt'))
model1.load_state_dict(torch.load('BEST/MI_best_model.ckpt', map_location=torch.device('cpu')))
model2.load_state_dict(torch.load('BEST/STTC_best_model.ckpt', map_location=torch.device('cpu')))
model3.load_state_dict(torch.load('BEST/CD_best_model.ckpt', map_location=torch.device('cpu')))
model4.load_state_dict(torch.load('BEST/HYP_best_model.ckpt', map_location=torch.device('cpu')))

models = [model1,model2, model3, model4]
# --------------- model list ---------------# --------------- model list ---------------# --------------- model list ---------------

# # Average -----------------------------------------------------------
class Average(nn.Module):
    def __init__(self, models):
        super(Average, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        output = []
        for model in self.models:
            output.append(model(x))
        output = torch.stack(output, dim=0)
        output = torch.mean(output, dim=0)
        return output
    
# Weight-----------------------------------------------------------
class Weight(nn.Module):
    def __init__(self, models, weights):
        super(Weight, self).__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights

    def forward(self, x):
        output = []
        for i, model in enumerate(self.models):
            output.append(self.weights[i] * model(x))
        output = torch.stack(output, dim=0)
        output = torch.sum(output, dim=0)
        return output
    
# # Soft Voting -----------------------------------------------------
class SoftVoting(nn.Module):
    def __init__(self, models):
        super(SoftVoting, self).__init__()
        self.models = nn.ModuleList(models)
    def forward(self, x):
        output = []
        for model in self.models:
            output.append(torch.sigmoid(model(x)).cpu().detach().numpy())
        output = np.array(output)
        predictions = np.mean(output, axis=0)
        return torch.tensor(predictions, dtype=torch.float32).unsqueeze(dim=1).to(x.device)

# # Hard Voting -----------------------------------------------------
class HardVoting(nn.Module):
    def __init__(self, models):
        super(HardVoting, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        output = []
        for model in self.models:
            output.append(torch.sigmoid(model(x)).cpu().detach().numpy().flatten())
        output = np.array(output)
        predictions = []
        for i in range(output.shape[1]):
            votes = Counter(output[:,i])
            prediction = max(votes, key=votes.get)
            predictions.append(prediction)
        return torch.tensor(predictions, dtype=torch.float32).unsqueeze(dim=1).to(x.device)


dataset_wrapper = PTBXLWrapper(args.batch_size)

meansig_stdsig = {
    1: (-0.00074335, 0.23675443),
    2: (-0.00071208, 0.23702329),
    3: (-0.00076585, 0.2383337),
    4: (-0.00075563, 0.23732454),
    5: (-0.00077779, 0.23828742),
    6: (-0.00074522, 0.23840146),
    7: (-0.00074464, 0.23864935),
    8: (-0.0007491, 0.23856031),
    9: (-0.00074286, 0.23686773),
    10: (-0.00078311, 0.23812694),
}

# Create an empty list to store the evaluation results for each fold
all_ave_results = []
all_wei_results = []
all_sv_results  = []
all_hv_results  = []

tasks = ['MI', 'STTC', 'CD', 'HYP']

for task in tasks:
    fold_ave_results = []
    fold_wei_results = []
    fold_sv_results  = []
    fold_hv_results  = []
    for i in range(1,11):
        print(task , "test fold = ", i)    
        meansig, stdsig = meansig_stdsig[i]
        args.task = task
        train_dataloader, val_dataloader, test_dataloader = dataset_wrapper.get_data_loaders(args , meansig = meansig, stdsig = stdsig, test_fold = i)
        print(torch.cuda.current_device())
        averaging = Average(models)
        weights = [0.4, 0.25, 0.3, 0.25]
        weighting  = Weight(models, weights)
        softvoting = SoftVoting(models)
        hardvoting = HardVoting(models)

        ave_results = evaluate(test_dataloader, averaging)
        wei_results = evaluate(test_dataloader, weighting)
        sv_results  = evaluate(test_dataloader, softvoting)
        hv_results  = evaluate(test_dataloader, hardvoting)     
        # print("ave", task, ave_results)  
        # print("hv", task, hv_results)  
        fold_ave_results.append(ave_results)
        fold_wei_results.append(wei_results)
        fold_sv_results.append(sv_results)
        fold_hv_results.append(hv_results)
    avg_ave_results = {}
    avg_wei_results = {}
    avg_sv_results = {}
    avg_hv_results = {}
    for key in fold_ave_results[0].keys():
        avg_ave_results[key] = np.mean([ave_results[key] for ave_results in fold_ave_results], axis=0)
    for key in fold_wei_results[0].keys():
        avg_wei_results[key] = np.mean([wei_results[key] for wei_results in fold_wei_results], axis=0)
    for key in fold_sv_results[0].keys():
        avg_sv_results[key] = np.mean([sv_results[key] for sv_results in fold_sv_results], axis=0)
    for key in fold_hv_results[0].keys():
        avg_hv_results[key] = np.mean([hv_results[key] for hv_results in fold_hv_results], axis=0)
    all_ave_results.append(fold_ave_results)
    all_wei_results.append(fold_wei_results)
    all_sv_results.append(fold_sv_results)
    all_hv_results.append(fold_hv_results)
all_ave_res = {}
all_wei_res = {}
all_sv_res = {}
all_hv_res = {}

for key in all_ave_results[0][0].keys():
    values = [fold_ave_results[i][key] for i in range(len(fold_ave_results))]
    mean_values = np.mean(values, axis=0)
    all_ave_res[key] = mean_values
for key in all_wei_results[0][0].keys():
    values = [fold_wei_results[i][key] for i in range(len(fold_ave_results))]
    mean_values = np.mean(values, axis=0)
    all_wei_res[key] = mean_values
for key in all_sv_results[0][0].keys():
    values = [fold_sv_results[i][key] for i in range(len(fold_ave_results))]
    mean_values = np.mean(values, axis=0)
    all_sv_res[key] = mean_values
for key in all_hv_results[0][0].keys():
    values = [fold_hv_results[i][key] for i in range(len(fold_ave_results))]
    mean_values = np.mean(values, axis=0)
    all_hv_res[key] = mean_values
    
# Compute the average of all results over all tasks and folds
print( "Averaging results: ===========================")
for key, value in all_ave_res.items():
    print(key, ":", value)
print( "Weighting results: ===========================")
for key, value in all_wei_res.items():
    print(key, ":", value)
print( "Softvoting results: ===========================")
for key, value in all_sv_res.items():
    print(key, ":", value)
print( "Hardvoting results: ===========================")
for key, value in all_hv_res.items():
    print(key, ":", value)
