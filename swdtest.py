import torch
import numpy as np
import ot

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 生成随机方向
def generate_random_directions(num_directions, dim, device):
    directions = torch.randn(num_directions, dim, device=device)
    directions /= torch.norm(directions, dim=1, keepdim=True)
    return directions

# 投影点
def project_points(points, direction):
    return torch.matmul(points, direction)

def wasserstein_distance(pp,qq):
    distm = torch.dist(pp.detach(),qq.detach(),p=2.0)
    return ot.emd2([],[],distm.cpu().numpy())

# 计算切片Wasserstein距离
def sliced_wasserstein_distance(p, q, num_directions=1000):
    p = torch.tensor(p, dtype=torch.float32, device=device)
    q = torch.tensor(q, dtype=torch.float32, device=device)
    
    dim = p.shape[1]
    directions = generate_random_directions(num_directions, dim, device)
    
    swd = 0.0
    for direction in directions:
        proj_p = project_points(p, direction)
        proj_q = project_points(q, direction)
        swd += wasserstein_distance(proj_p, proj_q)
    
    return swd / num_directions

# 示例数据
p = torch.tensor([[1, 2], [3, 4], [5, 6]])
q = torch.tensor([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]])

# 计算切片Wasserstein距离
swd = sliced_wasserstein_distance(p, q)
print("Sliced Wasserstein Distance:", swd)