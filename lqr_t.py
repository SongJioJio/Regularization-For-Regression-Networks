import numpy as np
from torchdiffeq import odeint
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import wandb
from sklearn import preprocessing
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier


#device = torch.device("cuda")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
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

lr = 1e-3
epochs = 1000

filename = 'lqr2.txt'
data = torch.tensor(np.loadtxt(filename,
                            delimiter=',',  # 用于分割各列值的字符
                            usecols=[0, 1, 2 ,3 ,4 ])  # 读取并使用第3列
                 )

data=data[torch.randperm(data.size(0))]
#minmax = preprocessing.MinMaxScaler()
#data_minmax = minmax.fit_transform(data)
#data_minmax = torch.Tensor(data_minmax)
indices = torch.tensor([0, 1, 2 ,3 ])
x0 = torch.index_select(data, 1, indices)  
u = data[:, 4]
#x0 = x0.to(device)
#u = u.to(device)


def train(epoch):
    loss = torch.zeros(1)
    #loss = loss.to(device)
    for idx in range(1, x0.size(0)+1):
        optimizer.zero_grad()
        #x = x0[idx-1].unsqueeze(dim=0)
        #output = net(x)
        output = net(x0[idx - 1])
        loss += loss_f(output, u[idx - 1])
        #loss += torch.sum((output - u[idx - 1]) ** 2)
    loss.backward()
    optimizer.step()
    print('Epoch:{} Loss {:.6f}'.format(epoch, loss.item()))

def train1(epoch):
    global regularization
    loss = torch.zeros(1)
    #loss = loss.to(device)
    for idx in range(x0.size(0)):
        optimizer.zero_grad()
        output = net(x0[idx])
        list1 = Find_Recent(x0, x0[idx], 2)
        id = []
        l = []
        for i in list1:
            l.append(i[1])
            id.append(i[0])
        c = []
        for j in range(len(id)):
            d = torch.dist(u[id[j]], u[idx], p=2) / torch.dist(x0[id[j]], x0[idx], p=np.inf)
            c.append(d)
        c2 = max(c)
        lambd = 0.1
        regularization = torch.zeros(1)
        #regularization = regularization.to(device)
        for i in range(100):
            a = 2*np.random.randint(0, 2, size=4)-1
            r = min(l)
            #r = min(l).cpu()
            x = np.random.uniform(0, r, size=4)
            #x = np.random.uniform(0, 0.05, size=3)
            deta_x = torch.tensor(a*x)
            #deta_x = deta_x.to(device)
            output1 = net(x0[idx] + deta_x)
            regularizer = loss_f(output, output1)
            m = torch.dist(output, output1, p=2) / torch.norm(deta_x)
            if m > c2:
                regularization += regularizer
            else:
                regularization = regularization
        regularization = regularization / 100
        loss += loss_f(output, u[idx]) + lambd * regularization
        #wandb.log({'epoch': epoch, 'idx': idx, 'regularization': regularization})
    loss.backward()
    optimizer.step()
    print('Epoch:{} Loss {:.6f} regularization:{}'.format(epoch, loss.item(), regularization))
    #wandb.log({'epoch': epoch, 'loss': loss})

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
def test1():
    global loss
    loss = torch.zeros(1)
    for idx in range(x0.size(0)):
        # x = x0[idx - 1].unsqueeze(dim=0)
        # output = new_net(x)
        attack = ProjectedGradientDescent(classifier, eps=0.1)
        #attack = FastGradientMethod(classifier, eps=0.1)
        #x_test = X_test[idx][np.newaxis, :]
        #X = attack.generate(x_test)
        X = attack.generate(x0[idx].numpy())
        x = torch.tensor(X)
        output =new_net(x)
        #u = -x ** 2 - 2 * np.tanh(x)
        #loss += torch.squeeze(loss_f(output, y_test[idx]), dim=0)
        loss += loss_f(output, u[idx])
    print('test Epoch:{} MSELoss:{:.6f}'.format(1, loss.data.item()/x0.size(0)))

def test():
    global loss
    loss = torch.zeros(1)
    for idx in range(x0.size(0)):
        # x = x0[idx - 1].unsqueeze(dim=0)
        # output = new_net(x)
        output = new_net(x0[idx])
        loss += loss_f(output, u[idx])
    print('test Epoch:{} MSELoss:{:.6f}'.format(1, loss.data.item()/x0.size(0)))




if __name__ == '__main__':
    #wandb.login()
    #config = dict(
    #    learning_rate=1e-2,
    #)
    #wandb.init(
    #    project="nn based control",
    #    name="train",
    #    config=config,
    #)
    
    net = Net()
    #net = net.to(device)
    loss_f = Criterion()
    #loss_f = loss_f.to(device)
    optimizer = optim.Adam(net.parameters(), lr)
    #wandb.watch(net, log="all")
    for epoch in range(1, epochs + 1):
        if epoch == 600:
            for p in optimizer.param_groups:
                p['lr'] *= 0.1
        train(epoch)
    
    #torch.save(net, 'tnet_43.pkl')
    torch.save(net.state_dict(), 'nlnet_params_ff.pkl')
    #wandb.save('tnet_params_91.pkl')


    new_net = Net()
    new_net.load_state_dict(torch.load('nlnet_params_ff.pkl'))
    optimizer = optim.Adam(new_net.parameters(), lr)

    classifier = PyTorchClassifier(
    model=new_net,
    loss=loss_f,
    optimizer=optimizer,
    input_shape=(4, 1),
    nb_classes=1000,
    device_type="cpu"
    )

    test2()
