
import os
import json
import sys

task = 'HYP'
# MI STTC CD HYP
with open('output_auc'+task+'.txt', 'w') as k:
    for i in range(2, 7):
        for j in range(1, 7):
            folder_path = '/home/jennifer/DA4ECG/Result/task2_' + \
                str(i)+'X'+str(j)+'/'

            for filename in os.listdir(folder_path):
                if filename.startswith('Task'+task):
                    if filename.endswith('.json'):

                        with open(os.path.join(folder_path, filename)) as f:
                            data = json.load(f)
                            epoch_loss = data['epoch_loss']
                            auc = data['auc']
                            auprc = data['auprc']
                            sys.stdout = k

                            # print('['+str(i)+'X'+str(j)+']')
                            # print(filename)
                            # print('epoch_loss', epoch_loss)  # Output: John
                            print('auc       ', auc)   # Output: 30
                            # print('auprc     ', auprc)  # Output: New York
                            sys.stdout = sys.__stdout__
                            print('['+str(i)+'X'+str(j)+']')
                            print('auc       ', auc)
