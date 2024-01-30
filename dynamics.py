import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torchdiffeq import odeint
import torch
import torch.nn as nn
from torch import sigmoid
import torch.nn.functional as F
import torch.optim as optim
import random
import wandb
from sklearn import preprocessing
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier

#device = torch.device("cuda")

filename = 'dynamics.txt'
X0 = np.loadtxt(filename,
               delimiter=',',  # 用于分割各列值的字符
               usecols=[0]
               )  # 读取并使用第1列和第2列
Y0 = np.loadtxt(filename,
               delimiter=',',  # 用于分割各列值的字符
               usecols=[1]
               )  # 读取并使用第3列

#train_data.shape -> (404, 13)
#test_data.shape -> (102, 13)
X_train, X_test, Y_train, Y_test = train_test_split(X0, Y0, test_size=0.2, random_state=10)

#数据标准化，减去平均值再除以标准差(测试数据也用训练数据的标准差)
#mean = X_train.mean(axis=0)#axis=0表示求每一列的平均，axis=1表示求每一行的平均
#X_train -= mean
#std = X_train.std(axis=0)
#X_train /= std
#得到的特征平均值为0,标准差为1
#X_test -= mean
#X_test /= std


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self._init_params()  # 执行初始化

    def forward(self, x):
        x = x.to(torch.float32)
        y = self.net(x)
        return y

    def _init_params(self):
        np.random.seed(3)
        for m in self.modules():
            if isinstance(m, nn.Linear):  # 对线性层的权重执行随机初始化
                m.weight.data = torch.randn(m.weight.data.shape)*0.01
                torch.nn.init.constant_(m.bias.data, 0)


lr = 1e-2
epochs = 1000

# 利用K折验证输入的数据
k = 4  # 将数据分为4个相同的折，每个折的第i-1个分区作为验证集
num_val = len(X_train) // k  # 每个分区大小(一定要整除)
mae_list = []

def train1(epoch):
    x_train = torch.tensor(X_train)
    t_train = torch.tensor(Y_train)
    x_train = x_train.unsqueeze(dim=-1)
    t_train = t_train.unsqueeze(dim=-1)
    loss = torch.zeros(1)
    #loss = loss.to(device)
    for idx in range(x_train.size(0)):
        optimizer.zero_grad()
        #x = x0[idx-1].unsqueeze(dim=0)
        #output = net(x)
        output = net(x_train[idx])
        loss += loss_f(output, t_train[idx])
        #loss += torch.sum((output - u[idx - 1]) ** 2)
    loss = loss / x_train.size(0)
    loss.backward()
    optimizer.step()
    print('Epoch:{} Loss {:.6f}'.format(epoch, loss.item()))

def train(epoch):
    x_train = torch.tensor(X_train)
    t_train = torch.tensor(Y_train)
    x_train = x_train.unsqueeze(dim=-1)
    t_train = t_train.unsqueeze(dim=-1)
    global regularization
    loss = torch.zeros(1)
    #loss = loss.to(device)
    for idx in range(x_train.size(0)):
        optimizer.zero_grad()
        #x_train[idx].unsqueeze(dim=0)
        output = net(x_train[idx])
        list1 = Find_Recent(x_train, x_train[idx], 4)
        id = []
        l = []
        for i in list1:
            l.append(i[1])
            id.append(i[0])
        c = []
        for j in range(len(id)):
            d = torch.dist(t_train[id[j]], t_train[idx], p=2) / torch.dist(x_train[id[j]], x_train[idx], p=np.inf)
            c.append(d)
        c2 = max(c)
        lambd = 0.1
        regularization = torch.zeros(1)
        # regularization = regularization.to(device)
        for i in range(1, 100):
            a = 2 * np.random.randint(0, 2, size=1) - 1
            r = min(l)
            x = np.random.uniform(0, radiu[r], size=1)
            deta_x = torch.tensor(a * x)
            output1 = net(x_train[idx] + deta_x)
            regularizer = loss_f(output, output1)
            m = torch.dist(output, output1, p=2) / torch.norm(deta_x)
            if m > c2:
                regularization += regularizer
            else:
                regularization = regularization
        regularization = regularization / 100
        loss += loss_f(output, t_train[idx]) + lambd * regularization
    loss = loss / x_train.size(0)
    loss.backward()
    optimizer.step()
    print('Epoch:{} Loss {:.6f} regularization:{}'.format(epoch, loss.item(), regularization))
    # wandb.log({'epoch': epoch, 'loss': loss})


def Find_Recent(x, x_, k):
    """
    找到点集中距离目标点(x, y)最近的k个点
    :param x: 点集的x坐标
    :param y: 点集的y坐标
    :param x_: 目标点的x坐标
    :param y_: 目标点的y坐标
    :param k: 距离目标点最近的k个值
    :return: k个与目标点最近的点的集合
    """
    list_stack_temp = []  # 建立一个空的栈
    for idx in range(x.size(0)):
        list_temp = []
        if not x[idx].equal(x_):
            distence = torch.dist(x[idx], x_, p=np.inf)
            #length = length.numpy()
            if len(list_stack_temp) < k:
                list_stack_temp.append([idx, distence])
                #print("临时栈中多了一组数据，目前有" + str(len(list_stack_temp)) + "组数据")

            else:
                for m in list_stack_temp:
                    list_temp.append(m[1])
                    #print("临时列表中有" + str(len(list_temp)) + "组数据")
                list_temp.append(distence)
                list_temp.sort()
                if distence != list_temp[-1]:
                    last_ = list_temp[-1]
                    for n in list_stack_temp:
                        if n[1] == last_:
                            list_stack_temp.remove(n)
                        else:
                            continue
                    list_stack_temp.append([idx, distence])
                else:
                    continue
        else:
            continue
    return list_stack_temp

class Criterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, y):
        out = (logits - y) ** 2
        return out


# 测试代码
def test():
    x_test = torch.tensor(X_test)
    y_test = torch.tensor(Y_test)
    x_test = x_test.unsqueeze(dim=-1)
    y_test = y_test.unsqueeze(dim=-1)
    global loss
    loss = torch.zeros(1)
    for idx in range(x_test.size(0)):
        # x = x0[idx - 1].unsqueeze(dim=0)
        # output = new_net(x)
        output = new_net(x_test[idx])
        loss += loss_f(output, y_test[idx])
    print('test Epoch:{} MSELoss:{:.6f}'.format(1, loss.data.item()/x_test.size(0)))


def test1():
    x_test = torch.tensor(X_test)
    y_test = torch.tensor(Y_test)
    x_test = x_test.unsqueeze(dim=-1)
    y_test = y_test.unsqueeze(dim=-1)
    global loss
    loss = torch.zeros(1)
    for idx in range(y_test.size(0)):
        # x = x0[idx - 1].unsqueeze(dim=0)
        # output = new_net(x)
        #attack = ProjectedGradientDescent(classifier, eps=0.1)
        attack = FastGradientMethod(classifier, eps=0.1)
        #x_test = X_test[idx][np.newaxis, :]
        #X = attack.generate(x_test)
        x_ = x_test[idx].numpy()
        X = attack.generate(x_)
        x = torch.tensor(X)
        output =new_net(x)
        loss += torch.squeeze(loss_f(output, y_test[idx]), dim=0)
        #loss += loss_f(output, y_test[idx])
    print('test Epoch:{} MSELoss:{:.6f}'.format(1, loss.data.item()/y_test.size(0)))


if __name__ == '__main__':
    net = Net()
    loss_f = Criterion()
    optimizer = optim.Adam(net.parameters(), lr)

    #for epoch in range(1, epochs + 1):
        #train1(epoch)

    #torch.save(net.state_dict(), 'ndnet_params_501.pkl')

    new_net = Net()
    new_net.eval()
    new_net.load_state_dict(torch.load('ndnet_params_50.pkl'))

    classifier = PyTorchClassifier(
        model=new_net,
        loss=loss_f,
        optimizer=optimizer,
        input_shape=(1, 1),
        nb_classes=1000,
        device_type="cpu"
    )
    test1()
