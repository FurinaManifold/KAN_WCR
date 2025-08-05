import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
    def abs(self, x, y):
        num_examples = x.size()[0]
        h = 1.0 / (x.size()[1] - 1.0)
        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)
        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]
        
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)
    




class RRMSE(object):
    def __init__(self, ):
        super(RRMSE, self).__init__()
        
    def __call__(self, x, y):
        num_examples = x.size()[0]
        
        norm = torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), 2 , 1)
        normy = torch.norm( y.view(num_examples,-1), 2 , 1)
        mean_norm = torch.mean((norm/normy))
        return mean_norm



def compute_error(drift, net_drift, diffusion, net_diffusion, device, data, error_type, intervalradius):
    interval=[-intervalradius,intervalradius]
    if error_type == 'uniform1d':
        n=2500
        x = torch.linspace(interval[0], interval[1], n, device=device).unsqueeze(1)
        exact_drift = drift(x) # n,1
        exact_diffusion = diffusion(x)[:,:,0] # n,1,1
        pred_drift = net_drift(x)# n,1
        pred_diffusion = torch.sqrt(net_diffusion(x)) # n,1
        error_drift = torch.mean((exact_drift - pred_drift) ** 2) / torch.mean(exact_drift ** 2)
        error_diffusion = torch.mean((exact_diffusion - pred_diffusion) ** 2) / torch.mean(exact_diffusion ** 2)
    if error_type == 'uniform2d':
        n=200
        x = torch.linspace(interval[0], interval[1], n, device=device)
        y = x
        X,Y = torch.meshgrid(x,y)
        data = torch.cat((X.unsqueeze(1),Y.unsqueeze(1)),dim  = -1).reshape(-1,2)
        exact_drift = drift(data) # n,2
        exact_diffusion = diffusion(data) #n,2,2
        
        pred_drift = net_drift(data)
        pred_diffusion = net_diffusion(data)

        error_drift = torch.mean((exact_drift - pred_drift) ** 2) / torch.mean(exact_drift ** 2)
        error_diffusion = torch.mean((exact_diffusion[:,0,0] - pred_diffusion[:,0]) ** 2) / torch.mean(exact_diffusion[:,0,0] ** 2) + torch.mean((exact_diffusion[:,1,1] - pred_diffusion[:,1]) ** 2) / torch.mean(exact_diffusion[:,1,1] ** 2)
    if error_type == 'uniform3d':
        n=50
        x = torch.linspace(interval[0], interval[1], n, device=device)
        y = x
        z = x
        X,Y,Z = torch.meshgrid(x,y,z)
        data = torch.cat((X.unsqueeze(1),Y.unsqueeze(1),Z.unsqueeze(1)),dim  = -1).reshape(-1,3)
        exact_drift = drift(data) # n,2
        exact_diffusion = diffusion(data) #n,2,2
        
        pred_drift = net_drift(data)
        pred_diffusion = net_diffusion(data)

        error_drift = torch.mean((exact_drift - pred_drift) ** 2) / torch.mean(exact_drift ** 2)
        error_diffusion = torch.mean((exact_diffusion[:,0,0] - pred_diffusion[:,0]) ** 2) / torch.mean(exact_diffusion[:,0,0] ** 2) + torch.mean((exact_diffusion[:,1,1] - pred_diffusion[:,1]) ** 2) / torch.mean(exact_diffusion[:,1,1] ** 2)

    
    
    return error_drift, error_diffusion

def drift_poly3(x):
    '''
    x:sample,dim
    out: sample,dim
    '''
    return x - x ** 3

def diffusion_knownnet(x):
    return  torch.ones_like(x)

def diffusion_constant(x):
    '''
    n,1
    '''
    diag = torch.eye(x.shape[-1]).cuda()
    #print(diag.unsqueeze(0).repeat(x.shape[0],1,1).shape)
    return  diag.unsqueeze(0).repeat(x.shape[0],1,1)

def diffusion_Chen(x):
    return 1.0138*torch.ones_like(x)

def drift_Chen(x):
    return 0.0225*torch.ones_like(x)+0.9638*x-0.0010*x**2-1.0035*x**3

def diffusion_Chen2(x):
    return torch.tensor([0.9863,0.9731]).repeat(x.shape[0],1).cuda()

def drift_Chen2(x):
    f1 = 0.0022*torch.ones(x.shape[0]).cuda()+0.9966*x[:,0]-0.0049*x[:,0]**2-0.9755*x[:,0]**3
    f2 = -0.0264*torch.ones(x.shape[0]).cuda()+0.9948*x[:,1]+0.0162*x[:,1]**2-0.9526*x[:,1]**3
    return torch.cat([f1.unsqueeze(1),f2.unsqueeze(1)],dim = 1).cuda()

if __name__ == '__main__':
    drift_error, diffusion_error = compute_error(drift = drift_poly3,
                                                net_drift=drift_Chen2,
                                                diffusion=diffusion_constant,
                                                net_diffusion=diffusion_Chen2,
                                                device='cuda',
                                                data=None,
                                                error_type='uniform2d',
                                                intervalradius = 1.0
                                                )
    print(drift_error)
    print(diffusion_error)



    
