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
from imblearn.over_sampling import RandomOverSampler,ADASYN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.scale1_conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=[3,31],
                padding=[1,0]),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.scale2_conv = torch.nn.Sequential(
            torch.nn.BatchNorm2d(1),
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=[5,31],
                padding=[2,0]),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.FNC_conv = torch.nn.Sequential(
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
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            # dropout=0.5
        )
        self.lstm2 = nn.LSTM(
            input_size=96,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            # dropout=0.5
        )

        self.fc = nn.Sequential(
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(8, 2),
        )

    def forward(self, x, x_FNC):
        x1 = self.scale1_conv(x)
        x2 = self.scale2_conv(x)
        x1=torch.transpose(x1, 1, 2)
        x1=x1.flatten(2).cpu()
        x2 = torch.transpose(x2, 1, 2)
        x2 = x2.flatten(2).cpu()
        x=torch.cat((x1,x2),2).to(device)
        out, (h_n, h_c) = self.lstm(x, None)
        out=torch.cat((out.cpu(),x.cpu()),2).to(device)
        out, (h_n, h_c) = self.lstm2(out, None)
        out = F.relu(out)
        out = out[:, -1, :]
        out_FNC = self.FNC_conv(x_FNC)
        print(out.shape)
        print(out_FNC.shape)
        x = self.fc(out)
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
            datanum=int(np.argwhere(np.array(sub_list)==sub)[0])+1
            if datanum<10:
                dn='00' + str(datanum)
            else:
                dn='0' + str(datanum)

            input_name = num + '_' + task

            if task == 'mot_1':
                label_name = num + '_func_' + num + '_task-mot_run-01_events'
                datapath = '/home/zmx/ds002311/ICA/' + 'ica_sub'+dn+'_timecourses_ica_s1_.nii'
            elif task == 'mot_2':
                label_name = num + '_func_' + num + '_task-mot_run-02_events'
                datapath = '/home/zmx/ds002311/ICA/' + 'ica_sub'+dn+'_timecourses_ica_s2_.nii'
            elif task == 'mot_3':
                label_name = num + '_func_' + num + '_task-mot_run-03_events'
                datapath = '/home/zmx/ds002311/ICA/' + 'ica_sub'+dn+'_timecourses_ica_s3_.nii'

            img = nib.load(datapath)
            img = np.array(img.get_fdata())
            data=img[12:,:]

            for i in range(24):
                time_course = data[16 * i:16 * i + 16,:]
                time_course = np.array(time_course)

                time_course = time_course.T # 沿时间轴标准化
                for ic in range(31):
                    for x in time_course[ic]:
                        x = float(x - np.mean(time_course[ic]))/np.std(time_course[ic])
                time_course = time_course.T
                Input.append(time_course)

            with open(path_label + label_name + '.tsv', 'rt') as csvfile:
                reader = csv.DictReader(csvfile, delimiter='\t')
                cond = [row['cond'] for row in reader]

            for i in range(len(cond)):
                if cond[i] == 'targ_easy':
                    cond[i] = 0
                elif cond[i] == 'targ_hard':
                    cond[i] = 0
                elif cond[i] == 'lure_hard':
                    cond[i] = 1
            del cond[24]  # 最后一段时间不全
            label.extend(cond)
            label = list(map(int, label))

    Input = np.array(Input)
    label = np.array(label)

    return Input, label, FNC


CNN = Cnn()
CNN.to(device)
epochs = 50
# 定义loss和optimizer
optimizer = optim.Adam(CNN.parameters(), lr=0.001, weight_decay=0.002)
criterion = nn.CrossEntropyLoss()

# 训练
train_correct = 0
train_total = 0
batch_size=16
data_x, data_y, data_FNC = dataReader([1, 3, 5, 6, 7, 8, 9, 10, 13, 14, 15, 18, 21, 22, 23], ['mot_1', 'mot_2', 'mot_3'])

from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y, train_FNC, test_FNC = train_test_split(data_x, data_y, data_FNC, test_size=0.2)
smo = ADASYN(random_state=42)  # 处理样本数量不对称
nsamples, nx, ny = train_x.shape
d2_train_dataset = train_x.reshape((nsamples, nx * ny))
train_x, train_y = smo.fit_sample(d2_train_dataset, train_y)
train_x = train_x.reshape(len(train_x), nx, ny)

smo = RandomOverSampler(random_state=42)
nsamples, nx, ny = test_x.shape
d2_train_dataset = test_x.reshape((nsamples, nx * ny))
test_x, test_y = smo.fit_sample(d2_train_dataset, test_y)
test_x = test_x.reshape(len(test_x), nx, ny)

test_x = torch.from_numpy(test_x)
test_x = torch.tensor(test_x, dtype=torch.float32)
test_y = torch.from_numpy(test_y)

for epoch in range(epochs):
    #CNN.train(mode=True)
    if epoch:
        train_x=np.array(train_x)
        train_y=np.array(train_y)

    state = np.random.get_state()  # 打乱顺序
    np.random.shuffle(train_x)
    np.random.set_state(state)
    np.random.shuffle(train_y)

    train_x = torch.from_numpy(train_x)
    train_x = torch.tensor(train_x, dtype=torch.float32)
    train_y = torch.from_numpy(train_y)
    train_y = torch.tensor(train_y, dtype=torch.long)
    for i in range(0, len(train_x) - batch_size, batch_size):
        loss_batch = 0
        for b_x, b_y in zip(train_x[i:i+batch_size], train_y[i:i+batch_size]):
            b_x = b_x.reshape(-1, 16, 31)
            b_x = b_x.reshape(-1, 1, 16, 31)
            b_y = b_y.reshape(-1)
            b_x, b_y = b_x.to(device), b_y.to(device)

            output = CNN(b_x)
            loss = criterion(output, b_y)
            loss_batch += loss

            _, predicted = torch.max(output.data, 1)
            train_total += b_y.size(0)
            train_correct += (predicted == b_y).sum().item()

        loss_batch = loss_batch / batch_size
        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step()

    TimeStr = time.asctime(time.localtime(time.time()))
    print('Epoch: {} --- {}'.format(epoch, TimeStr))
    print('Train Accuracy of the model: {} %'.format(100 * train_correct / train_total))
    print('Train Loss of the model: {}'.format(loss_batch))

    # 衰减的学习率
    for p in optimizer.param_groups:
            p['lr'] *= 0.95

    # 每个epoch测试一次查看loss和准确率
    #CNN.eval()
    with torch.no_grad():
        test_correct = 0
        test_total = 0
        test_avg_loss = 0

        for t_x, t_y in zip(test_x, test_y):
            t_x = t_x.reshape(-1, 16, 31)
            t_x = t_x.reshape(-1, 1, 16, 31)
            t_y = t_y.reshape(-1)
            t_x, t_y = t_x.to(device), t_y.to(device)
            test_output = CNN(t_x)
            test_avg_loss += criterion(test_output, t_y)
            _, predicted = torch.max(test_output.data, 1)
            test_total += t_y.size(0)
            test_correct += (predicted == t_y).sum().item()
        test_avg_loss = test_avg_loss / len(test_y)
        print('Test Accuracy of the model: {} %'.format(100 * test_correct / test_total))
        print('Test loss of the model: {}'.format(test_avg_loss))
