#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import csv
import torch
import numpy as np
import random
import torch.nn.functional as F
import time
import nibabel as nib
import matplotlib.pyplot as plt
from math import *
from torch import nn
from torch import optim
from torch.autograd import Variable
from imblearn.over_sampling import SMOTE,ADASYN,RandomOverSampler

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
            nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(8, 16, kernel_size=3, padding=0),
            nn.BatchNorm2d(16),
            torch.nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(28224, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 2)
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


train_task = ['mot_1','mot_2']
train_sub = [1,3,5,6,7,8,9,10,13,14,15,18,21,22,23]

train_x, train_y = dataReader(train_sub,train_task)

plt.ion() #开启interactive mode 成功的关键函数
plt.figure(1)
t_now = 0
t = []
loss_list = []
acc_list = []

CNN = Cnn()  
CNN.to(device)
epochs = 10
batch_size = 16
# 定义loss和optimizer
optimizer = optim.Adam(CNN.parameters(), lr=0.001, weight_decay=0.01)
criterion = nn.CrossEntropyLoss().to(device)

# 训练
correct = 0
total = 0
CNN.train(mode=True)
for epoch in range(epochs):
    if epoch % 5 == 0:  # 衰减的学习率
        for p in optimizer.param_groups:
            p['lr'] *= 0.9

    # smo = RandomOverSampler(random_state=42)    # 处理样本数量不对称
    # smo = ADASYN(random_state=42)
    smo = SMOTE(random_state=42)
    nsamples, nx, ny = train_x.shape
    d2_train_dataset = train_x.reshape((nsamples,nx*ny))
    train_x_smo, train_y_smo = smo.fit_sample(d2_train_dataset, train_y)
    train_x_smo = train_x_smo.reshape(len(train_x_smo), nx, ny)

    state = np.random.get_state()   # 打乱顺序
    np.random.shuffle(train_x_smo)
    np.random.set_state(state)
    np.random.shuffle(train_y_smo)

    train_x_smo = torch.from_numpy(train_x_smo)
    train_x_smo = train_x_smo.type(torch.FloatTensor)
    train_y_smo = torch.from_numpy(train_y_smo)

    for i in range(0, len(train_x_smo) - batch_size, batch_size):
        loss_batch = 0

        for b_x, b_y in zip(train_x_smo[i:i+batch_size],train_y_smo[i:i+batch_size]):
            b_x = b_x.reshape(-1, 90, 90)
            b_x = b_x.reshape(-1, 1, 90, 90)
            b_y = b_y.reshape(-1)
            b_x, b_y = b_x.to(device), b_y.to(device)
            output = CNN(b_x)
            loss = criterion(output,b_y)
            loss_batch += loss

            _, predicted = torch.max(output.data, 1)
            total += b_y.size(0)
            correct += (predicted == b_y).sum().item()
            # print('Target: {}'.format(b_y))
            # print('Output: {}'.format(torch.max(output, 1)[1]))

        loss_batch = loss_batch / batch_size
        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step()

        TimeStr = time.asctime(time.localtime(time.time()))
        print('Epoch: {} --- {}'.format(epoch, TimeStr))
        print('Train Accuracy of the model: {} %'.format(100 * correct / total))
        print('Train Loss of this batch: {}'.format(loss_batch))

        # if i % 5 == 0: # 隔一定数量的batch画图
        #     t.append(t_now)
        #     loss_list.append(loss_batch)
        #     acc_list.append(100 * correct / total)
        #     plt.subplot(2,1,1)
        #     plt.plot(t,loss_list,'-r')
        #     plt.title('loss',fontsize=10) 
        #     plt.tight_layout(h_pad=1)
        #     plt.subplot(2,1,2)
        #     plt.plot(t,acc_list,'-b')
        #     plt.title('acc',fontsize=10) 
        #     plt.draw()
        #     plt.pause(0.01)
        #     t_now += 5
            
        



# # 测试
CNN.eval()
with torch.no_grad():
    print('--------test--------')
    print('--------test--------')
    print('--------test--------')
    correct = 0
    total = 0
    test_task = ['mot_3']
    test_sub = [1,3,5,6,7,8,9,10,13,14,15,18,21,22,23]
    test_x, test_y = dataReader(test_sub,test_task)

    smo = RandomOverSampler(random_state=42)
    nsamples, nx, ny = test_x.shape
    d2_train_dataset = test_x.reshape((nsamples,nx*ny))
    test_x, test_y = smo.fit_sample(d2_train_dataset, test_y)
    test_x = test_x.reshape(len(test_x), nx, ny)

    test_x = torch.from_numpy(test_x)
    test_x = torch.tensor(test_x, dtype=torch.float32)
    test_y = torch.from_numpy(test_y)
    for t_x, t_y in zip(test_x, test_y):
        t_x = t_x.reshape(-1, 90, 90)
        t_x = t_x.reshape(-1, 1, 90, 90)
        t_y = t_y.reshape(-1)
        t_x, t_y = t_x.to(device), t_y.to(device)
        output = CNN(t_x)
        loss = criterion(output,t_y)
        _, predicted = torch.max(output.data, 1)
        total += t_y.size(0)
        correct += (predicted == t_y).sum().item()
        print('Test Accuracy of the model: {} %'.format(100 * correct / total))
        print('Target: {}'.format(t_y))
        print('Output: {}'.format(torch.max(output, 1)[1]))
        print('Test Loss of the model: {}'.format(loss))
