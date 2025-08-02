import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import ot
device = torch.device("cuda:0")
plt.rcParams['font.size'] = 24

# 固定随机种子
np.random.seed(0)
torch.manual_seed(0)

# -------------------------------
# 1. 数据生成（Euler-Maruyama 模拟 SDE）
# -------------------------------
N_initial = 100000       # 初始样本数量
true_sigma = 1.0        # 真正的 sigma

def true_a(x):
    x0=x[...,0].unsqueeze(-1)
    x1=x[...,1].unsqueeze(-1)
    return torch.cat([x0*(1-x0**2)*0.2+x1*(1+torch.sin(x0)),-x1+2*x0*(1-x0**2)*(1+torch.sin(x0))],dim = -1)     # 真正的漂移函数 a(x)

dt = 0.001              # 时间步长
T_end = 2.0             # 模拟终止时间
num_steps = int(T_end / dt)
# t_obs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0]  # 观测时刻
t_obs = [0.2,0.4,0.6,0.8,1.0,2.0]
step_indices = [int(t/dt) for t in t_obs]

# 初始分布：均值为0，标准差为0.3 的正态分布
X0 = torch.tensor(np.random.normal(0, 0.3, N_initial)).float()
Y0 = torch.tensor(np.random.normal(0, 0.3, N_initial)).float()
XY = torch.cat((X0.unsqueeze(1),Y0.unsqueeze(1)),dim  = -1).to(device)

# 用字典存储各个观测时刻的样本
data_dict = {t: None for t in t_obs}
rand_seriesX = []
rand_seriesY = []
for i in range(num_steps):
    rand_seriesX.append(np.random.randn(N_initial))
    rand_seriesY.append(np.random.randn(N_initial))
for i in range(num_steps):
    t_current = (i+1)*dt
    drift = true_a(XY)
    dW = np.sqrt(dt) * torch.cat((torch.tensor(rand_seriesX[i]).unsqueeze(1),torch.tensor(rand_seriesY[i]).unsqueeze(1)),dim  = -1).to(device)
    XY = XY + drift * dt + true_sigma * dW
    if i+1 in step_indices:
        t_val = round(t_current, 1)
        data_dict[t_val] = XY.clone()

# -------------------------------
# 2. PINNs 网络模型定义
# -------------------------------
# p_phi: 用于表示密度函数 p(x,t)
class DensityNN(nn.Module):
    def __init__(self):
        super(DensityNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 20),
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
            nn.Linear(2, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 2)   # 输出可以正负
        )
    def forward(self, x):
        return self.net(x)

p_net = DensityNN().to(device)
a_net = DriftNN().to(device)
# sigma 作为可学习参数，初始值设为 0.5
sigma0_param = nn.Parameter(torch.tensor([0.5], dtype=torch.float32, device=device), requires_grad=True)
sigma1_param = nn.Parameter(torch.tensor([0.5], dtype=torch.float32, device=device), requires_grad=True)
# -------------------------------
# 3. 优化器与训练设置
# -------------------------------
optimizer = optim.Adam(list(p_net.parameters()) + list(a_net.parameters()) + [sigma0_param,sigma1_param], lr=1e-2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
num_iterations = 10000         # 训练迭代次数
N_residual = 1000               # PDE 残差采样点数量

# 为 data loss 中积分部分预设梯形公式积分的网格（301 个点，积分区域仍用 [-3, 3]）
grid_x = torch.linspace(-2, 2, 201)
grid_y = grid_x
dx = grid_x[1] - grid_x[0]
# grid_x_tensor = torch.tensor(grid_x, dtype=torch.float32, device=device).unsqueeze(1)  # shape (301,1)
grid_x,grid_y = torch.meshgrid(grid_x,grid_y,indexing = 'ij')
grid_xy = torch.stack([grid_x.flatten(),grid_y.flatten()],dim = 1).to(device)
# 将观测数据转换为 torch tensor
obs_data = {}
for t_val in t_obs:
    obs_data[t_val] = data_dict[round(t_val, 1)].float()

# -------------------------------
# 4. 记录用于绘图的数据（每 500 次记录一次）
# -------------------------------
iters_record = []
loss_record = []         # 每 500 次迭代记录一次总 loss
a_error_history = []     # 每 500 次记录一次 drift 相对误差
sigma_error_history = [] # 每 500 次记录一次 sigma 相对误差

# 预设测试点，用于 drift 对比图，注意此处将区间设为 (-1.5, 1.5)
# x_test = torch.linspace(-1.2, 1.2, 200, device=device).unsqueeze(1)

# -------------------------------
# 5. 设置图表（不显示，而是保存到指定的 PNG 文件）
# -------------------------------
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
# 子图说明：
# axs[0,0] 训练 Loss 曲线
# axs[0,1] 相对误差曲线（drift 和 sigma）
# axs[1,0] 漂移函数对比图
# axs[1,1] 扩散系数对比图

# -------------------------------
# 6. 训练循环（使用 tqdm 进度条）
# -------------------------------
def train():
    update_interval = 500
    for it in tqdm(range(1, num_iterations+1), desc="Training"):
        optimizer.zero_grad()
        
        # PDE 残差 Loss
        x_r = torch.FloatTensor(N_residual, 1).uniform_(-2, 2).to(device)
        y_r = torch.FloatTensor(N_residual, 1).uniform_(-2, 2).to(device)
        t_r = torch.FloatTensor(N_residual, 1).uniform_(0.1, 1.0).to(device)
        x_r.requires_grad = True
        y_r.requires_grad = True
        t_r.requires_grad = True
        XY_t = torch.cat([x_r, y_r, t_r], dim=1)
        p_val = p_net(XY_t)
        # print(p_val.shape)
        p_t = torch.autograd.grad(p_val, t_r, grad_outputs=torch.ones_like(p_val), create_graph=True)[0]
        p_x = torch.autograd.grad(p_val, x_r, grad_outputs=torch.ones_like(p_val), create_graph=True)[0]
        p_y = torch.autograd.grad(p_val, y_r, grad_outputs=torch.ones_like(p_val), create_graph=True)[0]
        p_xx = torch.autograd.grad(p_x, x_r, grad_outputs=torch.ones_like(p_val), create_graph=True)[0]
        p_yy = torch.autograd.grad(p_y, y_r, grad_outputs=torch.ones_like(p_val), create_graph=True)[0]
        
        xy = torch.cat([x_r, y_r], dim=1)
        a_val = a_net(xy)
        ap = a_val * p_val
        ap_x = torch.autograd.grad(ap[:,0].unsqueeze(1), x_r, grad_outputs=torch.ones_like(p_val), create_graph=True)[0]
        ap_y = torch.autograd.grad(ap[:,1].unsqueeze(1), y_r, grad_outputs=torch.ones_like(p_val), create_graph=True)[0]
        
        residual = p_t + ap_x +ap_y - 0.5 *  (sigma0_param * p_xx+sigma1_param * p_yy)
        loss_pde = torch.mean(residual**2)
        
        # Data Loss：对每个观测时刻计算
        loss_data = 0.0
        for t_val in t_obs[:-1]:
            XY_obs = obs_data[t_val]
            n_obs = XY_obs.shape[0]
            t_obs_tensor = torch.full((n_obs, 1), float(t_val), dtype=torch.float32, device=device)
            XY_input = torch.cat([XY_obs, t_obs_tensor], dim=1)
            p_obs = p_net(XY_input)
            loss_log = - torch.mean(torch.log(p_obs + 1e-8))
            
            t_grid = t_val * torch.ones(grid_xy.shape[0]).unsqueeze(1).to(device)
            XY_grid = torch.cat([grid_xy, t_grid], dim=1)
            # print(XY_grid.shape)
            p_grid = p_net(XY_grid)
            integral = (dx/2) * (p_grid[0] + 2*torch.sum(p_grid[1:-1]) + p_grid[-1])
            
            loss_data += loss_log + integral
        
        loss_all = 2 * loss_pde + loss_data
        loss_all.backward()
        optimizer.step()
        
        # 每 update_interval 次记录一次指标并更新图像文件
        x_test = torch.linspace(-1, 1, 101, device=device)
        y_test = x_test 
        x_test,y_test = torch.meshgrid(x_test,y_test,indexing = 'ij')
        xy_test = torch.stack([x_test.flatten(),y_test.flatten()],dim = 1).to(device)
        if it % update_interval == 0:
            with torch.no_grad():
                a_pred_test = a_net(xy_test)
                a_true_test = true_a(xy_test) 
                rel_error_a = torch.mean(torch.norm(a_pred_test - a_true_test,dim = 1)**2) / torch.mean(torch.norm(a_true_test,dim = 1)**2)
                # rel_error_sigma = torch.abs(sigma_param - true_sigma) / true_sigma
            # iters_record.append(it)
            # loss_record.append(loss_all.item())
            # a_error_history.append(rel_error_a.item())
            # sigma_error_history.append(rel_error_sigma.item())
            # 更新并保存图像到 "training_progress.png"
            # update_plots(it)
    torch.save(a_net,'chen_2dgen_a_net.ckpt')
def W1calc():
    N_valid = 20000
    X0 = torch.tensor(np.random.normal(0, 0.3, N_valid)).float()
    Y0 = torch.tensor(np.random.normal(0, 0.3, N_valid)).float()
    XY = torch.cat((X0.unsqueeze(1),Y0.unsqueeze(1)),dim  = -1).to(device)
    rand_seriesX = []
    rand_seriesY = []
    for i in range(num_steps):
        rand_seriesX.append(np.random.randn(N_valid))
        rand_seriesY.append(np.random.randn(N_valid))
    for i in range(num_steps):
        t_current = (i+1)*dt
        drift = a_net(XY)
        dW = np.sqrt(dt) * torch.cat((torch.tensor(rand_seriesX[i]).unsqueeze(1),torch.tensor(rand_seriesY[i]).unsqueeze(1)),dim  = -1).to(device)
        XY = XY + drift * dt + true_sigma * dW.float()
    trueXY = XY.clone()

    X0 = torch.tensor(np.random.normal(0, 0.3, N_valid)).float()
    Y0 = torch.tensor(np.random.normal(0, 0.3, N_valid)).float()
    XY = torch.cat((X0.unsqueeze(1),Y0.unsqueeze(1)),dim  = -1).to(device)
    rand_seriesX = []
    rand_seriesY = []
    for i in range(num_steps):
        rand_seriesX.append(np.random.randn(N_valid))
        rand_seriesY.append(np.random.randn(N_valid))
    for i in range(num_steps):
        t_current = (i+1)*dt
        drift = true_a(XY)
        sigmadW = np.sqrt(dt) * torch.cat((sigma0_param * torch.tensor(rand_seriesX[i]).unsqueeze(1).to(device), sigma1_param * torch.tensor(rand_seriesY[i]).unsqueeze(1).to(device)),dim  = -1)
        XY = XY + drift * dt + sigmadW.float()

    # print(f"rel_err_a:{rel_error_a}")
    # print(f'sigma:{sigma0_param.item()},{sigma1_param.item()}')
    # predXY = XY.clone()
    # distm = torch.cdist(predXY,trueXY.detach(),p=2.0)
    # WD = ot.emd2([],[],distm.detach().cpu().numpy())
    # print(f'W1 distance:{WD}')
def draw():
    a_net = torch.load("chen_2dgen_a_net.ckpt").cuda()
    interval=[-1,1]
    n=50
    x = torch.linspace(interval[0], interval[1], n, device=device)
    y = x
    X,Y = torch.meshgrid(x,y)
    data = torch.cat((X.reshape(n,n,1),Y.reshape(n,n,1)),dim  = -1)
    exact_drift = torch.zeros(n,n,2)
    exact_diffusion = torch.zeros(n,n,2,2)
    pred_drift = torch.zeros(n,n,2)
    pred_diffusion = torch.zeros(n,n,2)

    #print("Xsize=",X.size())
    #print("Ysize=",Y.size())
    #print("datasize=",data.size())
    #print(data[0,0,:].unsqueeze(0))
    #print(drift_net(data[0,0,:].unsqueeze(0)))
    for i in range(n):
        for j in range(n):
            exact_drift[i,j,:]= true_a(data[i,j,:].unsqueeze(0))[0]
            pred_drift[i,j,:]= a_net(data[i,j,:].unsqueeze(0))[0]

    zlim = 3 
    levels = np.linspace(-zlim,zlim,51)
    plt.figure(figsize = (20,9))
    plt.subplot(1,2,1)
    plt.contourf(X.detach().cpu().numpy(), Y.detach().cpu().numpy(), pred_drift.detach().cpu().numpy()[...,0],80, cmap='rainbow',levels = levels)
    plt.xticks(np.linspace(-1,1,3))
    plt.yticks(np.linspace(-1,1,3))
    plt.title("pred_drift_X")
    cbar = plt.colorbar() 
    cbar.set_ticks(np.linspace(-zlim,zlim,5))
    plt.subplot(1,2,2)
    plt.contourf(X.detach().cpu().numpy(), Y.detach().cpu().numpy(), pred_drift.detach().cpu().numpy()[...,1],80, cmap='rainbow',levels = levels)
    plt.xticks(np.linspace(-1,1,3))
    plt.yticks(np.linspace(-1,1,3))
    plt.title("pred_drift_Y")
    cbar = plt.colorbar()
    cbar.set_ticks(np.linspace(-zlim,zlim,5))
    plt.savefig("chen/2d_gen_chen_drift.png")
    plt.close()

# train()
draw()
