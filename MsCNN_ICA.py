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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(1),
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=[3,31],
                padding=[1,0]),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(1),
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=[5,31],
                padding=[2,0]),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(1),
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=[7, 31],
                padding=[3, 0]),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(),
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(1),
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=[9, 31],
                padding=[4, 0]),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(12*128, 512),
            nn.Dropout(p=0.5),
            nn.Linear(512, 64),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        import torch
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x1 = torch.transpose(x1,1,2)
        x1 = x1.flatten(2).cpu()
        x2 = torch.transpose(x2, 1, 2)
        x2 = x2.flatten(2).cpu()
        x3 = torch.transpose(x3, 1, 2)
        x3 = x3.flatten(2).cpu()
        x4 = torch.transpose(x4, 1, 2)
        x4 = x4.flatten(2).cpu()
        x = torch.cat((x1,x2), 2)
        x = torch.cat((x, x3), 2)
        x = torch.cat((x, x4), 2)
        out = x.to(device)
        out = torch.flatten(out,start_dim=1)
        x = self.fc(out)
        return x, out


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
                time_course = data[16 * i + 1:16 * i + 13,:]
                time_course = np.array(time_course)

                # time_course = time_course.T # 沿时间轴标准化
                # for ic in range(31):
                #     for x in time_course[ic]:
                #         x = float(x - np.mean(time_course[ic]))/np.std(time_course[ic])
                # time_course = time_course.T
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

    return Input, label


CNN = Cnn()
CNN.to(device)
epochs = 50
# 定义loss和optimizer
optimizer = optim.Adam(CNN.parameters(), lr=0.001, weight_decay=0.05)
criterion = nn.CrossEntropyLoss()
scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=3,factor=0.5,threshold=1e-3,)

# 训练
train_correct = 0
train_total = 0
TP = 0
FP = 0
FN = 0
batch_size= 64
data_x, data_y = dataReader([1, 3, 5, 6, 7, 8, 9, 10, 13, 14, 15, 18, 21, 22, 23], ['mot_1', 'mot_2', 'mot_3'])

from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.05, random_state=RANDOM_SEED)
smo = SMOTE(random_state=RANDOM_SEED)  # 处理样本数量不对称
nsamples, nx, ny = train_x.shape
d2_train_dataset = train_x.reshape((nsamples, nx * ny))
train_x, train_y = smo.fit_sample(d2_train_dataset, train_y)
train_x = train_x.reshape(len(train_x), nx, ny)

smo = RandomOverSampler(random_state=RANDOM_SEED)
nsamples, nx, ny = test_x.shape
d2_train_dataset = test_x.reshape((nsamples, nx * ny))
test_x, test_y = smo.fit_sample(d2_train_dataset, test_y)
test_x = test_x.reshape(len(test_x), nx, ny)

test_x = torch.from_numpy(test_x)
test_x = torch.tensor(test_x, dtype=torch.float32)
test_y = torch.from_numpy(test_y)

plt.ion() # 开启interactive mode 绘制图形
plt.figure(1)
t_now = 0
t = []
loss_list = []
acc_list = []
pre_list = []
rec_list = []
if_legend = 1
from sklearn.manifold import TSNE
tsne=TSNE(n_components=2, random_state=RANDOM_SEED, init ='pca')

for epoch in range(epochs):
    CNN.train(mode=True)
    X_train_tsne=[]
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
    for i in range(0, len(train_x), batch_size):
        loss_batch = 0
        for b_x, b_y in zip(train_x[i:i+batch_size], train_y[i:i+batch_size]):
            b_x = b_x.reshape(-1, 12, 31)
            b_x = b_x.reshape(-1, 1, 12, 31)
            b_y = b_y.reshape(-1)
            b_x, b_y = b_x.to(device), b_y.to(device)
            
            output, temp = CNN(b_x)
            loss = criterion(output, b_y)
            loss_batch += loss

            _, predicted = torch.max(output.data, 1)
            train_total += b_y.size(0)
            train_correct += (predicted == b_y).sum().item()
            if predicted == 1 and b_y[0] == 1:
                TP += 1
            elif predicted == 1 and b_y[0] == 0:
                FP += 1
            elif predicted == 0 and b_y[0] == 1:
                FN += 1
            if epoch == epochs - 1:
                temp = temp.cpu()
                temp = temp.detach().numpy()
                X_train_tsne.append(temp)

        loss_batch = loss_batch / batch_size
        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step()

        if i % (5*batch_size) == 0: # 隔一定数量的batch画图
            t.append(t_now)
            loss_list.append(loss_batch)
            acc_list.append(100 * train_correct / train_total)
            pre_list.append(100 * TP / (TP + FP))
            rec_list.append(100 * TP / (TP + FN))
            plt.subplot(2,1,1)
            plt.plot(t,loss_list,'-r')
            plt.title('train loss',fontsize=15) 
            plt.xlabel(u"Number of train batches")
            plt.ylabel(u"loss")
            plt.tight_layout(h_pad=2)
            plt.subplot(2,1,2)
            plt.plot(t,acc_list,color='orange',label='accuracy')
            plt.plot(t,pre_list,color='green',label='precision')
            plt.plot(t,rec_list,color='blue',label='recall')
            if if_legend:
                plt.legend()
                if_legend = 0
            plt.title('Accuracy&Precision&Recall',fontsize=15) 
            plt.xlabel(u"Number of train batches")
            plt.ylabel(u"%")
            t_now += 5   
            plt.draw()
            plt.pause(0.01)     

    TimeStr = time.asctime(time.localtime(time.time()))
    print('Epoch: {} --- {} --- '.format(epoch, TimeStr))
    print('Train Accuracy of the model: {} %'.format(100 * train_correct / train_total))
    print('Train Loss of the model: {}'.format(loss_batch))
    print('Learning rate: {}'.format(optimizer.param_groups[0]['lr']))
    # 调整学习率
    scheduler.step(loss_batch)

    CNN.eval()
    with torch.no_grad():
        test_correct = 0
        test_total = 0
        test_avg_loss = 0
        X_test_tsne=[]

        for t_x, t_y in zip(test_x, test_y):
            t_x = t_x.reshape(-1, 12, 31)
            t_x = t_x.reshape(-1, 1, 12, 31)
            t_y = t_y.reshape(-1)
            t_x, t_y = t_x.to(device), t_y.to(device)
            test_output, temp2 = CNN(t_x)
            test_avg_loss += criterion(test_output, t_y)
            _, predicted = torch.max(test_output.data, 1)
            test_total += t_y.size(0)
            test_correct += (predicted == t_y).sum().item()

            if epoch == epochs - 1:
                temp2 = temp2.cpu()
                temp2 = temp2.detach().numpy()
                X_test_tsne.append(temp2)

        test_avg_loss = test_avg_loss / len(test_y)
        print('Test Accuracy of the model: {} %'.format(100 * test_correct / test_total))
        print('Test loss of the model: {}'.format(test_avg_loss))

'''

X_train_tsne = np.array(X_train_tsne)
train_size = X_train_tsne.shape[0]
X_train_tsne = X_train_tsne.reshape(X_train_tsne.shape[0], 1536)
X_test_tsne = np.array(X_test_tsne)
test_size = X_test_tsne.shape[0]
X_test_tsne = X_test_tsne.reshape(X_test_tsne.shape[0], 1536)
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
        if train_y[m]:
            lure_train_x.append(X_norm[m, 0])
            lure_train_y.append(X_norm[m, 1])
        else:
            targ_train_x.append(X_norm[m, 0])
            targ_train_y.append(X_norm[m, 1])
    else:
        if test_y[m - train_size]:
            lure_test_x.append(X_norm[m, 0])
            lure_test_y.append(X_norm[m, 1])
        else:
            targ_test_x.append(X_norm[m, 0])
            targ_test_y.append(X_norm[m, 1])

plt.subplot(1,2,1)
plt.title("train result")
plt.scatter(targ_train_x, targ_train_y, s=120,marker = ".", color='orange', label='targ', edgecolor='black',alpha=1) 
plt.scatter(lure_train_x, lure_train_y, s=120,marker = ".", color='green', label='lure', edgecolor='black',alpha=1) 
plt.legend()

plt.subplot(1,2,2)
plt.title("test result")
plt.scatter(targ_test_x, targ_test_y, s=120,marker = ".", color='orange', label='targ', edgecolor='black',alpha=1) 
plt.scatter(lure_test_x, lure_test_y, s=120,marker = ".", color='green', label='lure', edgecolor='black',alpha=1) 
plt.legend()

plt.draw()
plt.pause(0)
'''
