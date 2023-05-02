import os
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
import xgboost as xgb

cudnn.deterministic = True
cudnn.benchmark = False
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

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
# --------------- meansig stdsig list ---------------
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
# ---------------------------------------------------

dataset_wrapper = PTBXLWrapper(args.batch_size)

# Initialize lists to store results for each fold
val_accuracy_list = []
val_precision_list = []
val_recall_list = []
val_f1_list = []
val_auc_list = []
val_auprc_list = []

test_accuracy_list = []
test_precision_list = []
test_recall_list = []
test_f1_list = []
test_auc_list = []
test_auprc_list = []

models = [model1,model2, model3, model4]
tasks = ['MI', 'STTC', 'CD', 'HYP']

# Loop over all test folds

all_train_preds = []
all_train_labels = []

for i, task in enumerate(tasks):
    model = models[i]  # use specific model based on task
    model.eval()
    for test_fold in range(1, 3):
        meansig, stdsig = meansig_stdsig[test_fold]
        train_dataloader, val_dataloader, test_dataloader = dataset_wrapper.get_data_loaders(args,meansig=meansig, stdsig=stdsig, test_fold=test_fold)
        train_labels = dataset_wrapper.get_train_labels()
        val_labels = dataset_wrapper.get_val_labels()
        test_labels = dataset_wrapper.get_test_labels()

        # Step 1: Generate predictions for the selected model
        model_train_preds = []
        model_val_preds = []
        model_test_preds = []

        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Turn off gradient calculations to save memory and speed up computations
            # Iterate through the training dataset and generate predictions
            for x, y in train_dataloader:
                x = x.to(device)
                model_train_preds.append(model(x).detach().cpu().numpy())

            # Concatenate all the predictions generated for the training dataset and store them
            train_preds = np.concatenate(model_train_preds)

            # Generate predictions for the validation dataset
            for x, y in val_dataloader:
                x = x.to(device)
                model_val_preds.append(model(x).detach().cpu().numpy())
            val_preds = np.concatenate(model_val_preds)

            # Generate predictions for the test dataset
            for x, y in test_dataloader:
                x = x.to(device)
                model_test_preds.append(model(x).detach().cpu().numpy())
            test_preds = np.concatenate(model_test_preds)

        train_labels = np.array(train_labels)
        val_labels = np.array(val_labels)
        test_labels = np.array(test_labels)

        # Store the predictions and labels for the current fold
        all_train_preds.append(train_preds)
        all_train_labels.append(train_labels)

# Concatenate all the predictions and labels for all folds
all_train_preds = np.concatenate(all_train_preds)
all_train_labels = np.concatenate(all_train_labels)

# Create a dataset from all the model predictions and true labels
global_train_dataset = np.concatenate([all_train_preds, all_train_labels.reshape(-1, 1)], axis=1)

# Step 3: Train a logistic regression meta-classifier on top of the model predictions
# Train a logistic regression classifier on the global dataset
meta_classifier = LogisticRegression(random_state=42)
meta_classifier.fit(global_train_dataset[:, :-1], global_train_dataset[:, -1])

    
# = XGBoost = 
# meta_classifier = xgb.XGBClassifier(random_state=42)
# test_labels_8 = test_labels[:800]
# meta_classifier.fit(train_dataset, test_labels_8)

for test_fold in range(1, 11):
    print("test_fold : ", test_fold)
    meansig, stdsig = meansig_stdsig[test_fold]
    train_dataloader, val_dataloader, test_dataloader = dataset_wrapper.get_data_loaders(args,meansig=meansig, stdsig=stdsig, test_fold=test_fold)
    print(test_fold)
    train_labels = dataset_wrapper.get_train_labels()
    val_labels = dataset_wrapper.get_val_labels()
    test_labels = dataset_wrapper.get_test_labels()

    # Step 4: Evaluate the performance of the meta-classifier on the validation set
    val_data = []
    val_labels = []
    for batch in val_dataloader:
        data, labels = batch
        val_data.append(data.numpy())
        val_labels.append(labels.numpy())
    val_data = np.concatenate(val_data)
    val_data = val_data.reshape((val_data.shape[0], -1))  # Reshape to 2D
    val_preds_meta = meta_classifier.predict(val_data)
    val_prob_meta = meta_classifier.predict_proba(val_data)
    val_accuracy = accuracy_score(val_labels, val_preds_meta)
    val_f1 = f1_score(val_labels, val_preds_meta, average='macro')
    val_auc = roc_auc_score(val_labels, val_prob_meta[:, 1])
    val_auprc = average_precision_score(val_labels, val_prob_meta[:, 1])

    # Step 5: Generate final predictions on the test set
    test_data = []
    test_labels = []
    for batch in test_dataloader:
        data, labels = batch
        test_data.append(data.numpy())
        test_labels.append(labels.numpy())
    test_data = np.concatenate(test_data)
    test_data = test_data.reshape((test_data.shape[0], -1))  # Reshape to 2D
    test_labels = np.concatenate(test_labels)

    test_preds_meta = meta_classifier.predict(test_data)
    test_prob_meta = meta_classifier.predict_proba(test_data)
    test_accuracy = accuracy_score(test_labels, test_preds_meta)
    test_f1 = f1_score(test_labels, test_preds_meta, average='macro')
    test_auc = roc_auc_score(test_labels, test_prob_meta[:, 1])
    test_auprc = average_precision_score(test_labels, test_prob_meta[:, 1])

    # Store performance metrics for this fold
    val_accuracy_list.append(val_accuracy)
    val_f1_list.append(val_f1)
    val_auc_list.append(val_auc)
    val_auprc_list.append(val_auprc)
    test_accuracy_list.append(test_accuracy)
    test_f1_list.append(test_f1)
    test_auc_list.append(test_auc)
    test_auprc_list.append(test_auprc)


# Compute mean and standard deviation of performance metrics across all folds
print("Validation metrics:")
print(f"Accuracy:  {np.mean(val_accuracy_list):.3f} +/- {np.std(val_accuracy_list):.3f}")
print(f"F1-score:  {np.mean(val_f1_list):.3f} +/- {np.std(val_f1_list):.3f}")
print(f"AUC:       {np.mean(val_auc_list):.3f} +/- {np.std(val_auc_list):.3f}")
print(f"AUPRC:     {np.mean(val_auprc_list):.3f} +/- {np.std(val_auprc_list):.3f}")

print("")

print("Testing metrics:")
print(f"Accuracy:  {np.mean(test_accuracy_list):.3f} +/- {np.std(test_accuracy_list):.3f}")
print(f"F1-score:  {np.mean(test_f1_list):.3f} +/- {np.std(test_f1_list):.3f}")
print(f"AUC:       {np.mean(test_auc_list):.3f} +/- {np.std(test_auc_list):.3f}")
print(f"AUPRC:     {np.mean(test_auprc_list):.3f} +/- {np.std(test_auprc_list):.3f}")