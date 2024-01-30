import math
import time
import torch.optim as optim
import cvxpy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier

# Model parameters

l_bar = 2.0  # length of bar
M = 3.0  # [kg]
m = 1.0  # [kg]
g = 9.8  # [m/s^2]

nx = 4  # number of state
nu = 1  # number of input
Q = np.diag([0.0, 1.0, 1.0, 0.0])  # state cost matrix
R = np.diag([0.01])  # input cost matrix

T = 30  # Horizon length
delta_t = 0.1  # time tick
sim_time = 5.0  # simulation time [s]

show_animation = True


def main():
    loss = []
    x0 = np.array([
        [2.6],
        [0.0],
        [0.3],
        [0.0]
    ])

    x1 = np.copy(x0)
    x2 = np.copy(x0)
    time = 0.0

    while sim_time > time:
        time += delta_t

        # calc control input
        opt_x, opt_delta_x, opt_theta, opt_delta_theta, opt_input = mpc_control(x1)

        # get input
        u1 = opt_input[0]
        #print(f"{float(x[0]):.3f}, {float(x[1]):.3f},{math.degrees(x[2]):.3f},{float(x[3]):.3f},{float(u):.3f}")
        u2 = get_u2(model, x2)

        x2[2] = math.radians(x2[2])

        # simulate inverted pendulum cart
        x1 = simulation(x1, u1)
        x2 = simulation(x2, u2)

        if show_animation:
            err = loss_f(u1, u2)
            loss.append(err[0])

    print("Finish")
    print(f"x={float(x1[0, 0]):.2f} [m] , theta={math.degrees(x1[2, 0]):.2f} [deg]")
    print(f"x={float(x2[0, 0]):.2f} [m] , theta={math.degrees(x2[2, 0]):.2f} [deg]")
    print("test loss : ", sum(loss) / len(loss))
    if show_animation:
        t = np.arange(0.0, 5.1, 0.1)
        plt.title('loss')  # 折线图标题
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
        plt.rcParams['axes.unicode_minus'] = False  # 显示负号
        plt.plot(t, loss)
        plt.show()


def simulation(x, u):
    A, B = get_model_matrix()
    x = np.dot(A, x) + np.dot(B, u)

    return x


def mpc_control(x0):
    x = cvxpy.Variable((nx, T + 1))
    u = cvxpy.Variable((nu, T))

    A, B = get_model_matrix()

    cost = 0.0
    constr = []
    for t in range(T):
        cost += cvxpy.quad_form(x[:, t + 1], Q)
        cost += cvxpy.quad_form(u[:, t], R)
        constr += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t]]

    constr += [x[:, 0] == x0[:, 0]]
    prob = cvxpy.Problem(cvxpy.Minimize(cost), constr)

    start = time.time()
    prob.solve(verbose=False)
    #elapsed_time = time.time() - start
    #print(f"calc time:{elapsed_time:.6f} [sec]")

    if prob.status == cvxpy.OPTIMAL:
        ox = get_numpy_array_from_matrix(x.value[0, :])
        dx = get_numpy_array_from_matrix(x.value[1, :])
        theta = get_numpy_array_from_matrix(x.value[2, :])
        d_theta = get_numpy_array_from_matrix(x.value[3, :])

        ou = get_numpy_array_from_matrix(u.value[0, :])
    else:
        ox, dx, theta, d_theta, ou = None, None, None, None, None

    return ox, dx, theta, d_theta, ou


def get_numpy_array_from_matrix(x):
    """
    get build-in list from matrix
    """
    return np.array(x).flatten()


def get_model_matrix():
    A = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, m * g / M, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, g * (M + m) / (l_bar * M), 0.0]
    ])
    A = np.eye(nx) + delta_t * A

    B = np.array([
        [0.0],
        [1.0 / M],
        [0.0],
        [1.0 / (l_bar * M)]
    ])
    B = delta_t * B

    return A, B


# 获取输入u2
def get_u2(model, X):
    X[2] = math.degrees(X[2])
    x = torch.tensor(X.T)
    #attack = ProjectedGradientDescent(classifier, eps=0.01, max_iter=20)
    #attack = FastGradientMethod(classifier, eps=0.01)
    #x = attack.generate(x.numpy())
    #print(x)
    #x = torch.Tensor(x)
    u = model(x)
    u = u.detach().numpy()
    return u


def flatten(a):
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
    model.load_state_dict(torch.load('nmnet_params_e1.pkl'))
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
