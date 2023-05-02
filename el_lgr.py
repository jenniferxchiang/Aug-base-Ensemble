import os
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from model import *
from torch.backends import cudnn
from ptbxl_dataset import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

cudnn.deterministic = True
cudnn.benchmark = False

parser = argparse.ArgumentParser(description='ECG ensemble LogisticRegression')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--train_samp', type=int, default=1000)
parser.add_argument('--task', type=str, default='MI')
parser.add_argument('--batch_size', type=int, default=128)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# --------------- model list ---------------
model1 = resnet18(num_outputs=1).to(device)
model1.load_state_dict(torch.load('BEST/MI_best_model.ckpt'))
model2 = resnet18(num_outputs=1).to(device)
model2.load_state_dict(torch.load('BEST/STTC_best_model.ckpt'))
model3 = resnet18(num_outputs=1).to(device)
model3.load_state_dict(torch.load('BEST/CD_best_model.ckpt'))
model4 = resnet18(num_outputs=1).to(device)
model4.load_state_dict(torch.load('BEST/HYP_best_model.ckpt'))
# models = [model1,model2, model3, model4]
# --------------- meansig stdsig list ---------------
meansig_stdsig = {
    1: (-0.00074335, 0.23675443),2: (-0.00071208, 0.23702329),3: (-0.00076585, 0.2383337),
    4: (-0.00075563, 0.23732454),5: (-0.00077779, 0.23828742),6: (-0.00074522, 0.23840146),
    7: (-0.00074464, 0.23864935),8: (-0.0007491, 0.23856031), 9: (-0.00074286, 0.23686773),
    10: (-0.00078311, 0.23812694),}
# ---------------------------------------------------

dataset_wrapper = PTBXLWrapper(args.batch_size)

# Step 1: Generate predictions for each model on their corresponding tasks
models = [model1, model2, model3, model4]
tasks = ['MI', 'STTC', 'CD', 'HYP']
train_preds = {task: [] for task in tasks}
val_preds = {task: [] for task in tasks}
test_preds = {task: [] for task in tasks}

train_labels = []
val_labels = []
test_labels = []
fold_num = 10

for model, task in zip(models, tasks):
    model.eval()
    with torch.no_grad():
        for test_fold in range(1, fold_num+1): # 10-fold cross-validation
            meansig, stdsig = meansig_stdsig[test_fold]
            train_dataloader, val_dataloader, test_dataloader = dataset_wrapper.get_data_loaders(args,meansig=meansig, stdsig=stdsig, test_fold=test_fold)
            train_labels = dataset_wrapper.get_train_labels()
            val_labels   = dataset_wrapper.get_val_labels()
            test_labels  = dataset_wrapper.get_test_labels()
            
            train_preds_fold = []
            val_preds_fold = []
            test_preds_fold = []
            
            for x, y in train_dataloader:
                x = x.to(device)
                y = y.to(device)
                output = model(x)
                preds = output.argmax(dim=1)
                correct = preds.eq(y).sum().item()
                acc = correct / len(y)
                train_preds_fold.append(model(x).detach().cpu().numpy())
                print(f"Train Acc: {acc:.4f}")
                
            for x, y in val_dataloader:
                x = x.to(device)
                y = y.to(device)
                output = model(x)
                preds = output.argmax(dim=1)
                correct = preds.eq(y).sum().item()
                acc = correct / len(y)
                val_preds_fold.append(model(x).detach().cpu().numpy())
                print(f"Val Acc: {acc:.4f}")
                
                
            for x, y in test_dataloader:
                x = x.to(device)
                y = y.to(device)
                output = model(x)
                preds = output.argmax(dim=1)
                correct = preds.eq(y).sum().item()
                acc = correct / len(y)
                test_preds_fold.append(model(x).detach().cpu().numpy())
                print(f"Test Acc: {acc:.4f}")

            train_preds[task].append(np.concatenate(train_preds_fold))
            val_preds[task].append(np.concatenate(val_preds_fold))
            test_preds[task].append(np.concatenate(test_preds_fold))
            
# Step 2: Merge predictions for each task
merged_train_preds = [np.concatenate(train_preds[task]) for task in tasks]
merged_val_preds   = [np.concatenate(val_preds[task])   for task in tasks]
merged_test_preds  = [np.concatenate(test_preds[task])  for task in tasks]


# Step 3: Concatenate merged predictions for all tasks and train the global meta model
train_labels  = np.concatenate([train_labels for _ in range(fold_num)])
val_labels    = np.concatenate([val_labels   for _ in range(fold_num)])
test_labels   = np.concatenate([test_labels  for _ in range(fold_num)])

train_dataset = np.concatenate(merged_train_preds + [train_labels.reshape(-1, 1)], axis=1)
val_dataset   = np.concatenate(merged_val_preds + [val_labels.reshape(-1, 1)], axis=1)
# test_dataset  = np.concatenate(merged_test_preds + [test_labels.reshape(-1, 1)], axis=1)
# Get the minimum number of samples between merged_test_preds and test_labels
min_samples = min([pred.shape[0] for pred in merged_test_preds] + [test_labels.shape[0]])

# Truncate merged_test_preds and test_labels to have the same number of samples
merged_test_preds_truncated = [pred[:min_samples] for pred in merged_test_preds]
test_labels_truncated = test_labels[:min_samples]
# Concatenate the truncated arrays
test_dataset = np.concatenate(merged_test_preds_truncated + [test_labels_truncated.reshape(-1, 1)], axis=1)

meta_classifier = LogisticRegression(random_state=42)
meta_classifier.fit(train_dataset[:, :-1], train_dataset[:, -1])

# Step 4: Evaluate the performance of the meta-classifier on the validation set
val_preds_meta = meta_classifier.predict(val_dataset[:, :-1])
val_acc_meta = (val_preds_meta == val_dataset[:, -1]).mean()
print('Validation accuracy (meta-classifier): {:.4f}'.format(val_acc_meta))

# Step 5: Generate final predictions on the test set
test_preds_meta = meta_classifier.predict(test_dataset[:, :-1])
test_prob_meta = meta_classifier.predict_proba(test_dataset[:, :-1])
test_acc_meta = (test_preds_meta == test_dataset[:, -1]).mean()
test_f1 = f1_score(test_labels_truncated, test_preds_meta, average='macro')
test_auc = roc_auc_score(test_labels_truncated, test_preds_meta)
test_auprc = average_precision_score(test_labels_truncated, test_prob_meta[:, 1])
print('Test accuracy (meta-classifier): {:.4f}'.format(test_acc_meta))
print('Test f1 score (meta-classifier): {:.4f}'.format(test_f1))
print('Test roc auc  (meta-classifier): {:.4f}'.format(test_auc))
print('Test auprc    (meta-classifier): {:.4f}'.format(test_auprc))

