#--------------------------- edited ---------------------------#
#--------------------------- edited ---------------------------#
### PTB-XL data loading code adapted from https://physionet.org/content/ptb-xl/1.0.0/ ###

import ast
import wfdb
import pandas as pd
import random
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
import torch
import numpy as np
from tqdm import tqdm
PTBXL_PATH = '/home/jennifer/DA4ECG//ptb-xl/1.0.2'+'/'


class PTBXL(Dataset):
    def __init__(self, x_filelist, y, mean, std):
        super(PTBXL, self).__init__()
        self.x_filelist = x_filelist
        self.mean = mean
        self.std = std
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.x_filelist)

    def __getitem__(self, idx):
        x_path = self.x_filelist[idx]
        x = np.array(wfdb.rdsamp(x_path)[0])
        # Downsample to 250 Hz and chop off last 4 samples to get 2496 overall
        if x.shape[0] != 2496 and x.shape[0] == 5000:
            # pad
            x = x[::2, :]
            x = x[:-4]
        x = np.transpose(x, (1, 0)).astype(np.float32)
        x = (x - self.mean) / self.std
        y = self.y[idx]
        sample = (x, y)
        return sample


def calculate_mean_std(x_train_filelist):
    print("Calculating mean std of x train...")
    p1_sum = np.zeros(1)
    p2_sum = np.zeros(1)
    cnt = 0
    for p in tqdm(x_train_filelist):
        signal = np.array(wfdb.rdsamp(p)[0]).reshape(-1)
        p1_sum += signal.sum()
        p2_sum += np.power(signal, 2).sum()
        cnt += len(signal)
    s_mean = p1_sum / cnt
    s_std = np.sqrt(p2_sum / cnt - np.power(s_mean, 2))
    return s_mean, s_std


class PTBXLWrapper(object):

    def __init__(self, batch_size, num_workers=3):
        self.batch_size = batch_size
        self.num_workers = num_workers
    # load train labels
        self.train_labels = [] 
        self.val_labels = [] 
        self.test_labels = [] 

    
    
    def get_data_loaders(self, args, test_fold=10, meansig = -0.00074335, stdsig = 0.23675443):
        def load_raw_data(df, sampling_rate, path):
            path_list = []
            df_col = 'filename_lr' if sampling_rate == 100 else 'filename_hr'
            path_list = [path+f for f in getattr(df, df_col)]
            data = np.array(path_list, dtype=object)
            return data

        idxd = {'NORM': 0, 'MI': 1, 'STTC': 2, 'CD': 3, 'HYP': 4}

        def aggregate_diagnostic(y_dic):
            tmp = np.zeros(5)
            for key in y_dic.keys():
                if key in agg_df.index:
                    cls = agg_df.loc[key].diagnostic_class
                    tmp[idxd[cls]] = 1
            return tmp

        path = PTBXL_PATH
        sampling_rate = 500

        # load and convert annotation data
        Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load raw signal data
        X = load_raw_data(Y, sampling_rate, path)  # filelist

        # Load scp_statements.csv for diagnostic aggregation
        agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]

        # Apply diagnostic superclass
        Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

        # Split data into train and test
        # You can now modify the test_fold externally when you call the function
        # e.g. test_fold = 1 for the first fold, test_fold = 2 for the second fold, etc.
        # The default value for test_fold is 10, so if you don't modify it, it will use 10 as the test_fold value.
        # Train
        X_train = X[np.where(Y.strat_fold != test_fold)]
        y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
        y_train = np.stack(y_train, axis=0)

        # Test
        X_test = X[np.where(Y.strat_fold == test_fold)]
        y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass
        y_test = np.stack(y_test, axis=0)


        # Normalisation: follow PTB-XL demo code. Do zero mean
        # unit var normalisation across all leads, timesteps, and patients

        print(meansig, stdsig)

        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        rng = np.random.RandomState(args.seed)
        idxs = np.arange(len(y_train))
        rng.shuffle(idxs)

        train_samp = int(0.8*args.train_samp)
        val_samp = args.train_samp - train_samp

        train_idxs = idxs[:train_samp]
        val_idxs = idxs[train_samp:train_samp+val_samp]

        if args.task != 'all':
            task_idx = idxd[args.task]
            prevalence = np.mean(y_train[:, task_idx])
            self.weights = []
            for i in y_train[train_idxs][:, task_idx]:
                if i == 1:
                    self.weights.append(1-prevalence)
                else:
                    self.weights.append(prevalence)

            ft_train = PTBXL(X_train[train_idxs],
                             y_train[train_idxs][:, task_idx], meansig, stdsig)
            ft_val = PTBXL(X_train[val_idxs], y_train[val_idxs]
                           [:, task_idx], meansig, stdsig)
            ft_test = PTBXL(X_test, y_test[:, task_idx], meansig, stdsig)
            
            train_labels = y_train[train_idxs][:, task_idx]
            val_labels   = y_train[val_idxs][:, task_idx]
            test_labels  = y_test[:, task_idx]
            self.train_labels = torch.tensor(train_labels)
            self.val_labels = torch.tensor(val_labels)
            self.test_labels = torch.tensor(test_labels)
            print("Train:"+str(len(train_labels))," ==  Val:"+str(len(val_labels))," ==  Test:"+ str(len(test_labels)))

        else:
            ft_train = PTBXL(X_train[train_idxs],
                             y_train[train_idxs], meansig, stdsig)
            ft_val   = PTBXL(X_train[val_idxs],
                           y_train[val_idxs], meansig, stdsig)
            ft_test  = PTBXL(X_test, y_test, meansig, stdsig)

        train_loader = torch.utils.data.DataLoader(dataset=ft_train,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=self.num_workers)
        val_loader = torch.utils.data.DataLoader(dataset=ft_val,
                                                 batch_size=args.batch_size,
                                                 shuffle=True,
                                                 num_workers=self.num_workers)
        test_loader = torch.utils.data.DataLoader(dataset=ft_test,
                                                  batch_size=args.batch_size,
                                                  shuffle=True,
                                                  num_workers=self.num_workers)

        return train_loader, val_loader, test_loader
    
    def get_train_labels(self):
        return self.train_labels
    
    def get_val_labels(self):
        return self.val_labels
    
    def get_test_labels(self):
        return self.test_labels