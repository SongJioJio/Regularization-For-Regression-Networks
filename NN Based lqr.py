import math
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
import torch
import torch.nn as nn
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier


l_bar = 2.0  # length of bar
M = 1.0  # [kg]
m = 0.3  # [kg]
g = 9.8  # [m/s^2]

nx = 4  # number of state
nu = 1  # number of input
Q = np.diag([0.0, 1.0, 1.0, 0.0])  # state cost matrix。返回所给元素组成的对角矩阵
R = np.diag([0.01])  # input cost matrix

delta_t = 0.01  # time tick [s]
sim_time = 1.0  # simulation time [s]

show_animation = True  # 一个为真的变量
eps = 0.25


# 主函数√
def main():
    time = 0.0
    #px = []
    theta1 = []
    theta2 = []
    pu1 = []
    pu2 = []
    loss = []
    X0 = np.array([
        [1.9],
        [0.0],
        [0.2],
        [0.0]
    ])

    X2 = X0
    X1 = X0


    while sim_time > time:
        time += delta_t

        A, B = get_A_B()
        P = get_P(A, B, R, Q)
        K = get_K(A, B, R, P)
        u1 = get_u1(K, X1)
        u2 = get_u2(model, X2)
        X2[2] = math.radians(X2[2])

        if show_animation:
            theta1.append(math.degrees(X1[2]))
            theta2.append(math.degrees(X2[2]))
            pu1.append(float(u1))
            pu2.append(float(u2))
            err = loss_f(u1, u2)
            loss.append(err[0])

        X1 = A @ X1 + B @ u1
        X2 = A @ X2 + B @ u2
    print("Finish")
    print(f"x={float(X1[0]):.2f} [m] , theta={math.degrees(X1[2]):.2f} [deg]")
    print(f"x={float(X2[0]):.2f} [m] , theta={math.degrees(X2[2]):.2f} [deg]")
    print("test loss : ", sum(loss) / len(loss))
    if show_animation:
        t = np.arange(0.0, 1.0, 0.01)
        plt.title('loss')  # 折线图标题
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
        plt.rcParams['axes.unicode_minus'] = False  # 显示负号
        plt.plot(t, loss)
        plt.show()


# 获取p矩阵√
def get_P(A, B, R, Q):
    P = Q
    # Pn = A.T @ P @ A - A.T @ P @ B @ inv(R + B.T @ P @ B ) @ B.T @ P @ A +Q
    # print("Pn:",Pn)
    for i in range(150):
        Pn = A.T @ P @ A - A.T @ P @ B @ inv(R + B.T @ P @ B) @ B.T @ P @ A + Q
        # print("Pn:",Pn)
        # print("P:",P)
        if (abs(Pn - P)).max() < 0.01:
            break
        P = Pn

    return P


# 获取A,B√√√
def get_A_B():
    A = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, m * g / M, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, g * (M + m) / (l_bar * M), 0.0]
    ])

    B = np.array([
        [0.0],
        [1.0 / M],
        [0.0],
        [1.0 / (l_bar * M)]
    ])

    A = inv(np.eye(4) - A * 1 / 2 * delta_t) @ (np.eye(4) + A * 1 / 2 * delta_t)
    B = B * delta_t

    return A, B


#获取K矩阵
def get_K(A,B,R,P):
    K = inv(B.T @ P @ B + R) @(B.T @ P @ A)
    return K


# 获取输入u1
def get_u1(K, X):
    u = -1 * K @ X
    return u


# 获取输入u2
def get_u2(model, X):
    X[2] = math.degrees(X[2])
    x = torch.tensor(X.T)
    #attack = ProjectedGradientDescent(classifier, eps=0.01, max_iter=20)
    attack = FastGradientMethod(classifier, eps=0.01)
    x = attack.generate(x.numpy())
    x = torch.Tensor(x)
    u = model(x)
    u = u.detach().numpy()
    return u

# 获取输入u3
def get_u3(model, X):
    X[2] = math.degrees(X[2])
    x = torch.tensor(X.T)
    #attack = ProjectedGradientDescent(classifier, eps=0.001, max_iter=20)
    #attack = FastGradientMethod(classifier, eps=0.1)
    #x = attack.generate(x.numpy())
    #x = torch.Tensor(x)
    u = model(x)
    u = u.detach().numpy()
    return u


class Criterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, y):
        out = (logits - y) ** 2
        return out



## 8
def flatten(a):
    """
    将多维数组降为一维
    """
    return np.array(a).flatten()

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

    def forward(self, x):
        x = x.to(torch.float32)
        y = self.net(x)
        return y



class Criterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, y):
        out = (logits - y) ** 2
        return out


if __name__ == '__main__':
    model = Net()
    model.load_state_dict(torch.load('nlnet_params_ff.pkl'))
    model.eval()

    loss_f = Criterion()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    classifier = PyTorchClassifier(
        model=model,
        loss=loss_f,
        optimizer=optimizer,
        input_shape=(4, 1),
        nb_classes=1000,
        device_type="cpu"
    )
    main()
