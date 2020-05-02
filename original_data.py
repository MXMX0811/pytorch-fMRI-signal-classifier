import warnings 
warnings.filterwarnings('ignore')
import csv
import torch
import numpy as np
import random
import torch.nn.functional as F
import time
import nibabel as nib
from torch import nn
from torch import optim
from torch.autograd import Variable
from imblearn.over_sampling import RandomOverSampler,ADASYN,SMOTE
import matplotlib.pyplot as plt

RANDOM_SEED = 408     # 测试集0.05时
# RANDOM_SEED = 296     # 测试集0.1时


def dataReader(sub_list, task_list):
    Input = []
    label = []
    for task in task_list:
        for sub in sub_list:
            path_label = '/home/zmx/ds002311/event/'
            if sub < 10:
                num = 'sub-0' + str(sub)
            else:
                num = 'sub-' + str(sub)

            path_input = '/home/zmx/ds002311/FNC/' + num + '/' + task + '/'

            input_name = num + '_' + task

            if task == 'mot_1':
                label_name = num + '_func_' + num + '_task-mot_run-01_events'
            elif task == 'mot_2':
                label_name = num + '_func_' + num + '_task-mot_run-02_events'
            elif task == 'mot_3':
                label_name = num + '_func_' + num + '_task-mot_run-03_events'

            for i in range(24):
                img = np.loadtxt(open(path_input + input_name + '_' + str(i+1) + ".csv","rb"),delimiter=",",skiprows=0)
                Input.append(img)


            with open(path_label + label_name + '.tsv','rt') as csvfile:
                reader = csv.DictReader(csvfile, delimiter='\t')
                cond = [row['cond'] for row in reader]
            
            for i in range(len(cond)):
                if cond[i] == 'targ_easy':
                    cond[i] = 0
                elif cond[i] == 'targ_hard':
                    cond[i] = 0
                elif cond[i] == 'lure_hard':
                    cond[i] = 1
            label.extend(cond)
            label = list(map(int,label))
            del label[24]   # 最后一段时间不全

    Input = np.array(Input)
    label = np.array(label)

    return Input, label


data_x, data_y = dataReader([1, 3, 5, 6, 7, 8, 9, 10, 13, 14, 15, 18, 21, 22, 23], ['mot_1', 'mot_2', 'mot_3'])

from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.05, random_state=RANDOM_SEED)
original_train = train_x
original_y = train_y
smo = SMOTE(random_state=RANDOM_SEED)  # 处理样本数量不对称
nsamples, nx, ny = train_x.shape
d2_train_dataset = train_x.reshape((nsamples, nx * ny))
train_x, train_y = smo.fit_sample(d2_train_dataset, train_y)
train_x = train_x.reshape(len(train_x), nx, ny)
smote_train = train_x
smote_y = train_y

plt.ion() # 开启interactive mode 绘制图形
plt.figure(1)
t_now = 0
t = []
loss_list = []
acc_list = []
from sklearn.manifold import TSNE
tsne=TSNE(n_components=2, random_state=RANDOM_SEED, init ='pca')

X_train_tsne = original_train
print(X_train_tsne.shape)
train_size = X_train_tsne.shape[0]
X_train_tsne = X_train_tsne.reshape(X_train_tsne.shape[0], 90*90)
X_test_tsne = smote_train
print(X_test_tsne.shape)
test_size = X_test_tsne.shape[0]
X_test_tsne = X_test_tsne.reshape(X_test_tsne.shape[0], 90*90)
X_tsne = np.concatenate((X_train_tsne,X_test_tsne))
X_tsne = tsne.fit_transform(X_tsne)
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化

lure_train_x = []
lure_train_y = []
targ_train_x = []
targ_train_y = []
lure_test_x = []
lure_test_y = []
targ_test_x = []
targ_test_y = []
for m in range(X_norm.shape[0]):
    if m < train_size:
        if original_y[m]:
            lure_train_x.append(X_norm[m, 0])
            lure_train_y.append(X_norm[m, 1])
        else:
            targ_train_x.append(X_norm[m, 0])
            targ_train_y.append(X_norm[m, 1])
    else:
        if smote_y[m - train_size]:
            lure_test_x.append(X_norm[m, 0])
            lure_test_y.append(X_norm[m, 1])
        else:
            targ_test_x.append(X_norm[m, 0])
            targ_test_y.append(X_norm[m, 1])

plt.subplot(1,2,1)
plt.title("2D distribution of raw FNC data")
plt.scatter(targ_train_x, targ_train_y, s=120,marker = ".", color='orange', label='targ', edgecolor='black',alpha=1) 
plt.scatter(lure_train_x, lure_train_y, s=120,marker = ".", color='green', label='lure', edgecolor='black',alpha=1) 
plt.legend()

plt.subplot(1,2,2)
plt.title("2D distribution of FNC data with SMOTE")
plt.scatter(targ_test_x, targ_test_y, s=120,marker = ".", color='orange', label='targ', edgecolor='black',alpha=1) 
plt.scatter(lure_test_x, lure_test_y, s=120,marker = ".", color='green', label='lure', edgecolor='black',alpha=1) 
plt.legend()

plt.draw()
plt.pause(0)