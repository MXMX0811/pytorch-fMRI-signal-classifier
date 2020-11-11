#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

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
from imblearn.over_sampling import RandomOverSampler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class CnnLSTM(nn.Module):
    def __init__(self):
        super(CnnLSTM, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels=1,
                out_channels=4,
                kernel_size=3,
                padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(3),
            torch.nn.Conv3d(4, 8, kernel_size=7, padding=0),
            torch.nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size=22984,
            hidden_size=1024,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x,start_dim=1)
        x = x.reshape(16, 22984, 1)
        x = torch.transpose(x, 0, 2)
        x = torch.transpose(x, 1, 2)
        out, (h_n,h_c) = self.lstm(x, None)
        out = F.relu(out)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def dataReader(sub_list, task_list):
    Input = []
    label = []
    for task in task_list:
        for sub in sub_list:
            path_label = '/home/zmx/ds002311/event/'
            path_input = '/home/zmx/ds002311/preprocessed_4D/' + task + '/'
            if sub < 10:
                num = 'sub-0' + str(sub)
            else:
                num = 'sub-' + str(sub)

            input_name = num + '_' + task
            if task == 'mot_1':
                label_name = num + '_func_' + num + '_task-mot_run-01_events'
            elif task == 'mot_2':
                label_name = num + '_func_' + num + '_task-mot_run-02_events'
            elif task == 'mot_3':
                label_name = num + '_func_' + num + '_task-mot_run-03_events'

            img = nib.load(path_input + input_name + '.nii')
            img = np.array(img.get_fdata())
            template = nib.load('/home/zmx/fMRI/Template/area/AllBrain.nii')
            template = np.array(template.get_fdata())
            for i in range(61):
                for j in range(73):
                    for k in range(61):
                        if template[i][j][k]==0:
                            img[i][j][k] = np.zeros(405)

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
            del cond[24]   # 最后一段时间不全
            label.extend(cond)
            label = list(map(int,label))

            data = img[:,:,:,12:]   # 从第13个时间点开始，删除前12个时间点

            for i in range(24): # 最后一段时间不全
                Input.append(data[:,:,:,16*i:16*i+16])

    Input = np.array(Input)
    label = np.array(label)

    max_value = np.max(Input)  # 获得最大值
    min_value = np.min(Input)  # 获得最小值
    scalar = max_value - min_value  # 获得间隔数量
    Input = list(map(lambda x: x / scalar, Input)) # 归一化
    
    Input = np.array(Input)
    return Input, label


LSTM = CnnLSTM()  
LSTM.to(device)
epochs = 50
# 定义loss和optimizer
optimizer = optim.Adam(LSTM.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

# 训练
correct = 0
total = 0
for epoch in range(epochs):

    if epoch % 2 == 0:  # 衰减的学习率
        for p in optimizer.param_groups:
            p['lr'] *= 0.9
    #sub = [1,3,5,6,7,8,9,10,13,14,15,18,21,22,23]
    #task = ['mot_1','mot_2','mot_3']
    for task in [['mot_1'],['mot_2']]:
        for sub in [[1,3],[5,6]]:
            train_x, train_y = dataReader(sub,task)

            smo = RandomOverSampler(random_state=42)    # 处理样本数量不对称
            nsamples, nx, ny, nz, nt = train_x.shape
            d2_train_dataset = train_x.reshape((nsamples,nx*ny*nz*nt))
            train_x, train_y = smo.fit_sample(d2_train_dataset, train_y)
            train_x = train_x.reshape(len(train_x), nx, ny, nz, nt)

            state = np.random.get_state()   # 打乱顺序
            np.random.shuffle(train_x)
            np.random.set_state(state)
            np.random.shuffle(train_y)

            train_x = torch.from_numpy(train_x)
            train_x = torch.tensor(train_x, dtype=torch.float32)
            train_y = torch.from_numpy(train_y)

            for b_x, b_y in zip(train_x,train_y):
                b_x = b_x.reshape(-1, 61, 73, 61, 16)
                b_y = b_y.reshape(-1)
                b_x = torch.transpose(b_x, 3, 4)
                b_x = torch.transpose(b_x, 2, 3)
                b_x = torch.transpose(b_x, 1, 2)
                b_x = torch.transpose(b_x, 0, 1)
                b_x, b_y = b_x.to(device), b_y.to(device)
                output = LSTM(b_x)
                loss = criterion(output,b_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(output.data, 1)
                total += b_y.size(0)
                correct += (predicted == b_y).sum().item()

                TimeStr = time.asctime(time.localtime(time.time()))
                print('Epoch: {} --- {}'.format(epoch, TimeStr))
                print('Train Accuracy of the model: {} %'.format(100 * correct / total))
                print('Target: {}'.format(b_y))
                print('Output: {}'.format(torch.max(output, 1)[1]))
                print('Train Loss of the model: {}'.format(loss))



# # 测试
with torch.no_grad():
    print('--------test--------')
    print('--------test--------')
    print('--------test--------')
    correct = 0
    total = 0
    for task in [['mot_1']]:
        for sub in [10]:
            test_x, test_y = dataReader(sub,task)
            test_x = torch.from_numpy(test_x)
            test_x = torch.tensor(test_x, dtype=torch.float32)
            test_y = torch.from_numpy(test_y)
            for t_x, t_y in zip(test_x, test_y):
                t_x = t_x.reshape(-1, 61, 73, 61, 16)
                t_y = t_y.reshape(-1)
                t_x = torch.transpose(t_x, 3, 4)
                t_x = torch.transpose(t_x, 2, 3)
                t_x = torch.transpose(t_x, 1, 2)
                t_x = torch.transpose(t_x, 0, 1)
                t_x, t_y = t_x.to(device), t_y.to(device)
                output = LSTM(t_x)
                _, predicted = torch.max(output.data, 1)
                total += t_y.size(0)
                correct += (predicted == t_y).sum().item()
                print('Test Accuracy of the model: {} %'.format(100 * correct / total))
                print('Output: {}'.format(torch.max(output, 1)[1]))
