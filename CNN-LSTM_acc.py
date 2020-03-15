#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import csv
import torch
import numpy as np
import random
import torch.nn.functional as F
import time
from torch import nn
from torch import optim
from torch.autograd import Variable
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class CnnLSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim, bidirectional, n_layer, n_class):
        super(CnnLSTM, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=28,
                            out_channels=28,
                            kernel_size=3,
                            padding=1),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=28,
                            out_channels=28,
                            kernel_size=3,
                            padding=1),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=28,
                            out_channels=28,
                            kernel_size=3,
                            padding=1),
        )
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, bidirectional=bidirectional,
                            batch_first=True)
        if bidirectional:
            self.classifier = nn.Linear(hidden_dim*2, n_class)
        else:
            self.classifier = nn.Linear(hidden_dim, n_class)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(28, 16)   # 卷积方向不同需要先转置，卷积后再转置回来
        x = torch.t(x)
        x = x.reshape(-1, 16, 28)
        out, (h_n,h_c) = self.lstm(x, None)
        out = F.relu(out)
        out = out[:, -1, :]
        out = self.classifier(out)
        return out

def dataReader(sub, task):
    Input = []
    label = []
    path_label = '/home/zmx/ds002311/event/'
    for t in task:
        path_input = '/home/zmx/ds002311/preprocessed_4D/' + t + '/'
        for i in sub:
            if i < 10:
                num = 'sub-0' + str(i)
            else:
                num = 'sub-' + str(i)

            input_name = num + '_' + t
            if t == 'mot_1':
                label_name = num + '_func_' + num + '_task-mot_run-01_events'
            elif t == 'mot_2':
                label_name = num + '_func_' + num + '_task-mot_run-02_events'
            elif t == 'mot_3':
                label_name = num + '_func_' + num + '_task-mot_run-03_events'
            else:
                label_name = num + '_func_' + num + '_task-' + t + '_events'

            data = csv.reader(open(path_input + input_name + '.csv'))
            with open(path_label + label_name + '.tsv','rt') as csvfile:
                reader = csv.DictReader(csvfile, delimiter='\t')
                label.extend([row['acc'] for row in reader])

            label = list(map(int,label))
            data = list(data)
            del data[0:13]   # 从第13个时间点开始，删除表头以及前12个时间点
            del label[24]   # 最后一段时间不全
            float_data = []
            for each in data:
                each_line=list(map(lambda x: float(x), each))
                float_data.append(each_line)

            for i in range(24): # 最后一段时间不全
                Input.append(float_data[16*i:16*i+16])
    Input = np.array(Input)
    label = np.array(label)

    max_value = np.max(Input)  # 获得最大值
    min_value = np.min(Input)  # 获得最小值
    scalar = max_value - min_value  # 获得间隔数量
    Input = list(map(lambda x: x / scalar, Input)) # 归一化
    
    Input = np.array(Input)
    return Input, label

#sub = [1,3,5,6,7,8,9,10,13,14,15,18,21,22,23]
#task = ['loc','mot_1','mot_2','mot_3','prememory','postmemory']
train_x, train_y = dataReader([1,3,6],['mot_1'])
test_x, test_y = dataReader([6],['mot_1'])
test_x = torch.from_numpy(test_x)
test_x = torch.tensor(test_x, dtype=torch.float32)
test_y = torch.from_numpy(test_y)

smo = SMOTE(random_state=42)    # SMOTE处理样本数量不对称
nsamples, nx, ny = train_x.shape
d2_train_dataset = train_x.reshape((nsamples,nx*ny))
train_x_smo, train_y_smo = smo.fit_sample(d2_train_dataset, train_y)
train_x_smo = train_x_smo.reshape(len(train_x_smo), nx, ny)

epochs = 100
sequence_length = 16
input_size = 28
# 28个脑区，双向，序列长度为16，LSTM网络层数2层，二分类
LSTM = CnnLSTM(28, 256, True, 2, 2)  
LSTM.to(device)

# 定义loss和optimizer
optimizer = optim.Adam(LSTM.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# 训练
correct = 0
total = 0
for epoch in range(epochs):
    state = np.random.get_state()
    np.random.shuffle(train_x_smo)
    np.random.set_state(state)
    np.random.shuffle(train_y_smo)

    data_x = torch.from_numpy(train_x_smo)
    data_x = torch.tensor(data_x, dtype=torch.float32)
    data_y = torch.from_numpy(train_y_smo)

    i = 1
    

    for b_x, b_y in zip(data_x,data_y):
        b_x = b_x.reshape(sequence_length, input_size)
        b_y = b_y.reshape(-1)
        b_x = torch.t(b_x)
        b_x = b_x.reshape(-1, input_size, sequence_length)
        b_x, b_y = b_x.to(device), b_y.to(device)
        output = LSTM(b_x)
        loss = criterion(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(output.data, 1)
        total += b_y.size(0)
        correct += (predicted == b_y).sum().item()
        if i % 20 == 0:
            TimeStr = time.asctime(time.localtime(time.time()))
            print('Epoch: {} --- {}'.format(epoch, TimeStr))
            print('Train Accuracy of the model: {} %'.format(100 * correct / total))
            print('Train Loss of the model: {}'.format(loss))
        i += 1


# 测试
with torch.no_grad():
    print('--------test--------')
    print('--------test--------')
    print('--------test--------')
    correct = 0
    total = 0
    for t_x, t_y in zip(test_x, test_y):
        t_x = t_x.reshape(sequence_length, input_size)
        t_x = torch.t(t_x)
        t_x = t_x.reshape(-1, input_size, sequence_length)
        t_y = t_y.reshape(-1)
        t_x, t_y = t_x.to(device), t_y.to(device)
        output = LSTM(t_x)
        loss = criterion(output,t_y)
        _, predicted = torch.max(output.data, 1)
        total += t_y.size(0)
        correct += (predicted == t_y).sum().item()
        print('Test Accuracy of the model: {} %'.format(100 * correct / total))
        print('Test Loss of the model: {}'.format(loss))