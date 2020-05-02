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

class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,
                padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(8, 16, kernel_size=3, padding=0),
            torch.nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(28224, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x,start_dim=1)
        x = self.fc(x)
        return x

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
                label.extend([row['acc'] for row in reader])
            
            label = list(map(int,label))
            del label[24]   # 最后一段时间不全

    Input = np.array(Input)
    label = np.array(label)

    return Input, label


CNN = Cnn()  
CNN.to(device)
epochs = 5
# 定义loss和optimizer
optimizer = optim.Adam(CNN.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss()

# 训练
correct = 0
total = 0
for epoch in range(epochs):

    if epoch % 5 == 0:  # 衰减的学习率
        for p in optimizer.param_groups:
            p['lr'] *= 0.9
    #sub = [1,3,5,6,7,8,9,10,13,14,15,18,21,22,23]
    #task = ['mot_1','mot_2','mot_3']
    for task in [['mot_1'],['mot_2']]:
        for sub in [[1,3,5,6,7],[8,9,10,13,14],[15,18,21,22,23]]:
            # 一次5人
            train_x, train_y = dataReader(sub,task)

            smo = RandomOverSampler(random_state=42)    # 处理样本数量不对称
            nsamples, nx, ny = train_x.shape
            d2_train_dataset = train_x.reshape((nsamples,nx*ny))
            train_x, train_y = smo.fit_sample(d2_train_dataset, train_y)
            train_x = train_x.reshape(len(train_x), nx, ny)

            state = np.random.get_state()   # 打乱顺序
            np.random.shuffle(train_x)
            np.random.set_state(state)
            np.random.shuffle(train_y)

            train_x = torch.from_numpy(train_x)
            train_x = torch.tensor(train_x, dtype=torch.float32)
            train_y = torch.from_numpy(train_y)
            for b_x, b_y in zip(train_x,train_y):
                b_x = b_x.reshape(-1, 90, 90)
                b_x = b_x.reshape(-1, 1, 90, 90)
                b_y = b_y.reshape(-1)
                b_x, b_y = b_x.to(device), b_y.to(device)
                output = CNN(b_x)
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
    for task in [['mot_3']]:
        for sub in [[1,3,5,6,7],[8,9,10,13,14],[15,18,21,22,23]]:
            test_x, test_y = dataReader(sub,task)
            test_x = torch.from_numpy(test_x)
            test_x = torch.tensor(test_x, dtype=torch.float32)
            test_y = torch.from_numpy(test_y)
            for t_x, t_y in zip(test_x, test_y):
                t_x = t_x.reshape(-1, 90, 90)
                t_x = t_x.reshape(-1, 1, 90, 90)
                t_y = t_y.reshape(-1)
                t_x, t_y = t_x.to(device), t_y.to(device)
                output = CNN(t_x)
                _, predicted = torch.max(output.data, 1)
                total += t_y.size(0)
                correct += (predicted == t_y).sum().item()
                print('Test Accuracy of the model: {} %'.format(100 * correct / total))
                print('Target: {}'.format(t_y))
                print('Output: {}'.format(torch.max(output, 1)[1]))