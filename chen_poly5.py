import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import ot
import math
import json
plt.rcParams['font.size'] = 24

# 固定随机种子
np.random.seed(0)
torch.manual_seed(0)

# -------------------------------
# 1. 数据生成（Euler-Maruyama 模拟 SDE）
# -------------------------------
N_initial = 10000       # 初始样本数量
true_sigma = 1.0        # 真正的 sigma

def true_a(x):
    return - 2 * x + 8 * x ** 3 - 6.6 * x ** 5     # 真正的漂移函数 a(x)

dt = 0.001              # 时间步长
T_end = 2.0             # 模拟终止时间
num_steps = int(T_end / dt)
t_obs = t_obs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0]  # 观测时刻
step_indices = [int(t/dt) for t in t_obs]

# 初始分布：均值为0，标准差为0.3 的正态分布
X0 = np.random.normal(0, 0.3, N_initial)
X = X0.copy()

# 用字典存储各个观测时刻的样本
data_dict = {t: None for t in t_obs}
rand_series = []
for i in range(num_steps):
    rand_series.append(np.random.randn(N_initial))
for i in range(num_steps):
    t_current = (i+1)*dt
    drift = true_a(X)
    dW = np.sqrt(dt) * rand_series[i]
    X = X + drift * dt + true_sigma * dW
    if i+1 in step_indices:
        t_val = round(t_current, 1)
        data_dict[t_val] = X.copy()

# -------------------------------
# 2. PINNs 网络模型定义
# -------------------------------
# p_phi: 用于表示密度函数 p(x,t)
class DensityNN(nn.Module):
    def __init__(self):
        super(DensityNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1),
            nn.Softplus()   # 保证输出非负
        )
    def forward(self, x):
        return self.net(x)

# a_theta: 用于表示漂移函数 a(x)
class DriftNN(nn.Module):
    def __init__(self):
        super(DriftNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)   # 输出可以正负
        )
    def forward(self, x):
        return self.net(x)

device = torch.device("cuda:0")
p_net = DensityNN().to(device)
a_net = DriftNN().to(device)
# sigma 作为可学习参数，初始值设为 0.5
sigma_param = nn.Parameter(torch.tensor([0.5], dtype=torch.float32, device=device), requires_grad=True)

# -------------------------------
# 3. 优化器与训练设置
# -------------------------------
optimizer = optim.Adam(list(p_net.parameters()) + list(a_net.parameters()) + [sigma_param], lr=1e-2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
num_iterations = 10000         # 训练迭代次数
N_residual = 1000               # PDE 残差采样点数量

# 为 data loss 中积分部分预设梯形公式积分的网格（301 个点，积分区域仍用 [-3, 3]）
grid_x = np.linspace(-3, 3, 301)
dx = grid_x[1] - grid_x[0]
grid_x_tensor = torch.tensor(grid_x, dtype=torch.float32, device=device).unsqueeze(1)  # shape (301,1)

# 将观测数据转换为 torch tensor
obs_data = {}
for t_val in t_obs:
    data_array = data_dict[round(t_val, 1)]
    obs_data[t_val] = torch.tensor(data_array, dtype=torch.float32, device=device).unsqueeze(1)

# -------------------------------
# 4. 记录用于绘图的数据（每 500 次记录一次）
# -------------------------------
iters_record = []
loss_record = []         # 每 500 次迭代记录一次总 loss
a_error_history = []     # 每 500 次记录一次 drift 相对误差
sigma_error_history = [] # 每 500 次记录一次 sigma 相对误差

# 预设测试点，用于 drift 对比图，注意此处将区间设为 (-1.5, 1.5)
x_test = torch.linspace(-1.2, 1.2, 200, device=device).unsqueeze(1)

# -------------------------------
# 5. 设置图表（不显示，而是保存到指定的 PNG 文件）
# -------------------------------
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
# 子图说明：
# axs[0,0] 训练 Loss 曲线
# axs[0,1] 相对误差曲线（drift 和 sigma）
# axs[1,0] 漂移函数对比图
# axs[1,1] 扩散系数对比图

def update_plots(iteration):
    for ax in axs.flat:
        ax.cla()  # 清空子图内容
    
    # (1) 训练 Loss 曲线
    axs[0,0].plot(iters_record, loss_record, 'b-')
    axs[0,0].set_xlabel("Iteration")
    axs[0,0].set_ylabel("Total Loss")
    axs[0,0].set_title("Training Loss Curve")
    
    # (2) 相对误差曲线
    axs[0,1].plot(iters_record, a_error_history, 'r-', label="a(x) Relative Error")
    axs[0,1].plot(iters_record, sigma_error_history, 'g-', label="sigma Relative Error")
    axs[0,1].set_xlabel("Iteration")
    axs[0,1].set_ylabel("Relative Error")
    axs[0,1].set_title("Test Error Curve")
    axs[0,1].legend()
    
    # (3) 漂移函数对比图（在 (-1.5, 1.5) 区间内）
    a_pred = a_net(x_test).detach().cpu().numpy()
    x_test_np = x_test.cpu().numpy()
    a_true_np = x_test_np - x_test_np**3
    axs[1,0].plot(x_test_np, a_true_np, 'k-', label="True a(x)")
    axs[1,0].plot(x_test_np, a_pred, 'r--', label="Learned a(x)")
    axs[1,0].set_xlabel("x")
    axs[1,0].set_ylabel("a(x)")
    axs[1,0].set_title("Drift Function Comparison")
    axs[1,0].legend()
    
    # (4) 扩散系数对比图（横坐标设为 (-1.5, 1.5)）
    x_line = np.linspace(-1.2, 1.2, 100)
    sigma_learned = sigma_param.item()
    axs[1,1].plot(x_line, sigma_learned*np.ones_like(x_line), 'b--', label="Learned σ")
    axs[1,1].plot(x_line, true_sigma*np.ones_like(x_line), 'k-', label="True σ")
    axs[1,1].set_xlabel("x")
    axs[1,1].set_ylabel("σ")
    axs[1,1].set_title("Diffusion Coefficient Comparison")
    axs[1,1].legend()
    
    fig.suptitle(f"Iteration {iteration}")
    fig.tight_layout()
    # 保存图像到指定文件，每次更新都会覆盖之前的图片
    fig.savefig("training_progress.png")

# -------------------------------
# 6. 训练循环（使用 tqdm 进度条）
# -------------------------------
def train():
    update_interval = 10
    for it in tqdm(range(1, num_iterations+1), desc="Training"):
        optimizer.zero_grad()
        
        # PDE 残差 Loss
        x_r = torch.FloatTensor(N_residual, 1).uniform_(-3, 3).to(device)
        t_r = torch.FloatTensor(N_residual, 1).uniform_(0.1, 1.0).to(device)
        x_r.requires_grad = True
        t_r.requires_grad = True
        X_t = torch.cat([x_r, t_r], dim=1)
        p_val = p_net(X_t)
        p_t = torch.autograd.grad(p_val, t_r, grad_outputs=torch.ones_like(p_val), create_graph=True)[0]
        p_x = torch.autograd.grad(p_val, x_r, grad_outputs=torch.ones_like(p_val), create_graph=True)[0]
        p_xx = torch.autograd.grad(p_x, x_r, grad_outputs=torch.ones_like(p_val), create_graph=True)[0]
        
        a_val = a_net(x_r)
        ap = a_val * p_val
        ap_x = torch.autograd.grad(ap, x_r, grad_outputs=torch.ones_like(p_val), create_graph=True)[0]
        
        sigma_val = sigma_param
        residual = p_t + ap_x - 0.5 * sigma_val * p_xx
        loss_pde = torch.mean(residual**2)
        
        # Data Loss：对每个观测时刻计算
        loss_data = 0.0
        for t_val in t_obs[:-1]:
            X_obs = obs_data[t_val]
            n_obs = X_obs.shape[0]
            t_obs_tensor = torch.full((n_obs, 1), float(t_val), dtype=torch.float32, device=device)
            X_input = torch.cat([X_obs, t_obs_tensor], dim=1)
            p_obs = p_net(X_input)
            loss_log = - torch.mean(torch.log(p_obs + 1e-8))
            
            t_grid = torch.full_like(grid_x_tensor, float(t_val))
            X_grid = torch.cat([grid_x_tensor, t_grid], dim=1)
            p_grid = p_net(X_grid).squeeze()
            integral = (dx/2) * (p_grid[0] + 2*torch.sum(p_grid[1:-1]) + p_grid[-1])
            
            loss_data += loss_log + integral
        
        loss_all = 2 * loss_pde + loss_data
        loss_all.backward()
        optimizer.step()
        
        # 每 update_interval 次记录一次指标并更新图像文件
        if it % update_interval == 0:
            with torch.no_grad():
                a_pred_test = a_net(x_test)
                a_true_test = true_a(x_test)
                rel_error_a = torch.mean((a_pred_test - a_true_test)**2) / torch.mean(a_true_test**2)
                # rel_error_sigma = torch.abs(sigma_param - true_sigma) / true_sigma
            # iters_record.append(it)
            # loss_record.append(loss_all.item())
            a_error_history.append(rel_error_a.item())
            # sigma_error_history.append(rel_error_sigma.item())
            # 更新并保存图像到 "training_progress.png"
            # update_plots(it)
    torch.save(p_net,'chen_poly5_p_net.ckpt')
    torch.save(a_net,'chen_poly5_a_net.ckpt')
    with open('chen_poly5_training_a_error','w') as f:
        f.write(json.dumps(a_error_history))

def W1calc():
    N_valid = 10000
    X0 = np.random.normal(0, 0.3, N_valid)
    rand_series = []
    for i in range(num_steps):
        rand_series.append(np.random.randn(N_valid))
    X = torch.tensor(X0, dtype=torch.float32, device=device)
    for i in range(num_steps):
        t_current = (i+1)*dt
        drift = true_a(X)
        dW = torch.tensor(np.sqrt(dt) * rand_series[i], dtype=torch.float32, device=device)
        # print(X.shape,drift.shape,dW.shape)
        X = X + drift * dt + true_sigma * dW
    trueX = X.unsqueeze(1)

    X0 = np.random.normal(0, 0.3, N_valid)
    rand_series = []
    for i in range(num_steps):
        rand_series.append(np.random.randn(N_valid))
    X = torch.tensor(X0, dtype=torch.float32, device=device)
    for i in range(num_steps):
        t_current = (i+1)*dt
        drift = a_net(X.unsqueeze(1)).squeeze(1)
        dW = torch.tensor(np.sqrt(dt) * rand_series[i], dtype=torch.float32, device=device)
        # print(X.shape,drift.shape,dW.shape)
        X = X + drift * dt + sigma_param.item() * dW

    # print(f"rel_err_a:{rel_error_a}")
    # print(f'sigma:{sigma_param.item()}')
    # predX = X.unsqueeze(1)
    # distm = torch.cdist(predX,trueX.detach(),p=2.0)
    # WD = ot.emd2([],[],distm.detach().cpu().numpy())
    # print(f'W1 distance:{WD}')
def draw():
    a_net = torch.load("chen_poly5_a_net.ckpt").cuda()
    interval=[-1.2,1.2]
    n=100
    x = torch.linspace(interval[0], interval[1], n, device=device).unsqueeze(1)
    exact_drift = true_a(x) # n,1
    with torch.no_grad():
        pred_drift = a_net(x)# n,1
    maxy = math.ceil(abs(exact_drift[0].item()))
    plt.figure(figsize = (8,7))
    plt.plot(x.cpu().detach().numpy(), exact_drift.cpu().detach().numpy(), label="true", linewidth = 3,color = 'red')
    plt.plot(x.cpu().detach().numpy(), pred_drift.cpu().detach().numpy(), label="predicted", linewidth = 3,color = 'blue')
    plt.xticks(np.linspace(interval[0],interval[1],3))
    plt.yticks(np.linspace(-maxy,maxy,3))
    plt.ylim(-maxy-0.5,maxy+0.5)
    plt.xlabel("x")
    plt.ylabel("drift")
    plt.legend()
    plt.show()
    plt.savefig("chen/1d_poly5_chen_drift.png")
    plt.close()

# train()
draw()


