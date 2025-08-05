import torch
import torch.nn as nn
import numpy as np
import math
from torch import optim
from collections import OrderedDict
import time
import psutil
import os
import scipy.io
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from utils.utils import set_seed
import itertools
import logging
logger = logging.getLogger(__name__)
import wandb
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 24
import seaborn as sns
import yaml
from utils.getconfig import getcfg 
from data.GenerateData_fun import DataSet
from scipy.stats import entropy
import ot

def typechange(x):
    res = torch.zeros()
    return torch.einsum("ij,jk->ijk")

def plot_distributioncompare(drift_fun,drift_net,diffusion_fun,diffusion_net,cfg_data):
    dt = cfg_data['dt'] # 生成数据的dt
    t = torch.linspace(0,cfg_data['plotT'],cfg_data['nt']).cuda()# 生成数据的时间轴
    dim = cfg_data['dim'] #问题维数
    samples = cfg_data['sample'] if dim == 1 else cfg_data['plotsample']
    initialization=torch.normal(mean=0., std=cfg_data['sigma_init'], size=(samples, dim))
    if dim == 1:
        dataset_pred = DataSet(t, dt=dt, samples_num=samples, dim=cfg_data['dim'], drift_fun=drift_net, diffusion_fun=lambda x:diffusion_net(x).unsqueeze(-1),
                        initialization=initialization)
        dataset_exact = DataSet(t, dt=dt, samples_num=samples, dim=cfg_data['dim'], drift_fun=drift_fun, diffusion_fun=diffusion_fun,
                                initialization=initialization)
        print("Data loaded\n")
        preddata = dataset_pred.get_data(plot_hist=True,logname="gendata_distribution_dim",pltxlim = cfg_data['pltxlim'],reset_seed = True)
        exactdata = dataset_exact.get_data(plot_hist=True,logname="exactdata_distribution_dim",pltxlim = cfg_data['pltxlim'],reset_seed = True)
        print("Finished drawing distributions\n")
        # distm = torch.cdist(preddata[-1,:,:].detach(),exactdata[-1,:,:].detach(),p=2.0)
        # WD = ot.emd2([],[],distm.cpu().numpy())
        # print("W1 distance calculated\n")
        # try:
        #     wandb.log({"Wasserstein_dist":WD})
        #     print("W1 distance reported\n")
        # except Exception as ex:
        #     print(ex)
            
    elif dim == 2:
        dataset_pred = DataSet(t, dt=dt, samples_num=samples, dim=dim, drift_fun=drift_net, diffusion_fun=lambda x:torch.diag_embed(diffusion_net(x)),
                        initialization=initialization)
        dataset_exact = DataSet(t, dt=dt, samples_num=samples, dim=dim, drift_fun=drift_fun, diffusion_fun=diffusion_fun,
                                initialization=initialization)
        print("Data loaded\n")
        preddata = dataset_pred.get_data(plot_hist=False,logname="2dgendata_plot", plot2d_hist=True,dottype = 'b.',pltxlim = cfg_data['pltxlim'],reset_seed = True)
        exactdata = dataset_exact.get_data(plot_hist=False,logname="exactdata_distribution_dim",plot2d_hist=True,pltxlim = cfg_data['pltxlim'],reset_seed = True)
        print("Finished drawing distributions\n")
        distm = torch.cdist(preddata[-1,:,:].detach(),exactdata[-1,:,:].detach(),p=2.0)
        WD = ot.emd2([],[],distm.cpu().numpy())
        print("W1 distance calculated\n")
        try:
            wandb.log({"Wasserstein_dist":WD})
            print("W1 distance reported\n")
        except Exception as ex:
            print(ex)


def plot_distribution1(drift_fun,drift_net,diffusion_fun,diffusion_net,cfg_data):
    dt = cfg_data['dt'] # 生成数据的dt
    t = torch.linspace(0,cfg_data['T'],cfg_data['nt']).cuda()# 生成数据的时间轴
    dim = cfg_data['dim'] #问题维数
    samples = cfg_data['sample'] #生成轨道数量
    dataset = DataSet(t, dt=dt, samples_num=samples, dim=dim, drift_fun=drift_fun, diffusion_fun=diffusion_fun,
                      initialization=torch.normal(mean=0., std=cfg_data['sigma_init'], size=(samples, dim)))
    if dim == 1:
        data = dataset.get_data(plot_hist=True,logname="1ddata_plot_1",pltxlim = cfg_data['pltxlim']) # t, sample_num ,dim
    elif dim == 2:
        data = dataset.get_data(plot2d_hist=True,logname="2ddata_plot_1",histlim = 100,pltxlim = cfg_data['pltxlim'])
    
    
def plot_orbitcompare(drift_fun,drift_net,diffusion_fun,diffusion_net,cfg_data):
    dt = cfg_data['dt'] # 生成数据的dt
    t = torch.linspace(0,cfg_data['plotT'],cfg_data['plotnt']).cuda()# 生成数据的时间轴
    samples = cfg_data['plotsample'] #生成轨道数量
    initialization=torch.normal(mean=0., std=cfg_data['sigma_init'], size=(samples, 1))
    dataset_pred = DataSet(t, dt=dt, samples_num=samples, dim=1, drift_fun=drift_net, diffusion_fun=lambda x:torch.sqrt(diffusion_net(x)).unsqueeze(-1),
                       initialization=initialization)
    # dataset_pred = DataSet(t, dt=dt, samples_num=samples, dim=1, drift_fun=drift_fun, diffusion_fun=diffusion_fun,
    #                     initialization=initialization)
    dataset_exact = DataSet(t, dt=dt, samples_num=samples, dim=1, drift_fun=drift_fun, diffusion_fun=diffusion_fun,
                        initialization=initialization)
    plt.figure(figsize=(8,7))
    for i in range(samples-1):
        plt.plot(t.cpu().detach().numpy(),dataset_pred.get_data(reset_seed = True)[:,i,0].cpu().detach().numpy(),color='blue', linewidth = 2)
        plt.plot(t.cpu().detach().numpy(),dataset_exact.get_data(reset_seed = True)[:,i,0].cpu().detach().numpy(),color='red', linewidth = 2)
    plt.plot(t.cpu().detach().numpy(),dataset_pred.get_data(reset_seed = True)[:,samples-1,0].cpu().detach().numpy(),color='blue',label = "predicted orbits", linewidth = 2)
    plt.plot(t.cpu().detach().numpy(),dataset_exact.get_data(reset_seed = True)[:,samples-1,0].cpu().detach().numpy(),color='red',label = "true orbits", linewidth = 2)
    plt.xticks(np.linspace(0,cfg_data['plotT'],3))
    plt.yticks(np.linspace(-3,3,3))
    plt.xlim(0,2)
    plt.ylim(-3,3)
    plt.xlabel("t")
    plt.ylabel("x")
    plt.legend(loc = 'upper right')
    try:
        wandb.log({'orbits': wandb.Image(plt)})
    except:
        pass
    plt.close()

def plot_error(drift_fun,drift_net,diffusion_fun,diffusion_net,device,plot_type,cfg_data): 
    plot_distribution1(drift_fun,drift_net,diffusion_fun,diffusion_net,cfg_data)
    if plot_type == '1d':
        plot_1d(drift_fun,drift_net,diffusion_fun,diffusion_net,device,cfg_data['intervalradius'])
        print("Finished drawing drift and diffusion results\n")
        plot_orbitcompare(drift_fun,drift_net,diffusion_fun,diffusion_net,cfg_data)
        print("Finished drawing orbits\n")
        plot_distributioncompare(drift_fun,drift_net,diffusion_fun,diffusion_net,cfg_data)
    elif plot_type == '2d':
        plot_2d(drift_fun,drift_net,diffusion_fun,diffusion_net,device,cfg_data['intervalradius'],cfg_data['model'])
        print("Finished drawing drift and diffusion results\n")
        plot_distributioncompare(drift_fun,drift_net,diffusion_fun,diffusion_net,cfg_data)
        

def plot_1d(drift_fun,drift_net,diffusion_fun,diffusion_net,device,intervalradius):
    interval=[-intervalradius,intervalradius]
    n=100
    x = torch.linspace(interval[0], interval[1], n, device=device).unsqueeze(1)
    exact_drift = drift_fun(x) # n,1
    exact_diffusion = diffusion_fun(x)[...,0] # n,1
    with torch.no_grad():
        pred_drift = drift_net(x)# n,1
        pred_diffusion = torch.sqrt(diffusion_net(x)) # n,1
    maxy = math.ceil(abs(exact_drift[0].item()))
    plt.figure(figsize = (8,7))
    plt.plot(x.cpu().detach().numpy(), exact_drift.cpu().detach().numpy(), label="true", linewidth = 3,color = 'red')
    plt.plot(x.cpu().detach().numpy(), pred_drift.cpu().detach().numpy(), label="predicted", linewidth = 3,color = 'blue')
    plt.xticks(np.linspace(interval[0],interval[1],3))
    plt.yticks(np.linspace(-maxy,maxy,3))
    plt.xlim(interval[0]-0.3,interval[1]+0.3)
    plt.ylim(-maxy-0.5,maxy+0.5)
    plt.xlabel("x")
    plt.ylabel("drift")
    plt.legend(loc = 'upper right')
    plt.show()
    img = wandb.Image(plt)
    try:
        wandb.log({'Image_drift': img})
    except:
        pass
    plt.savefig("save_files/Image_drift.png")
    plt.close()
    plt.figure(figsize = (8,7))
    plt.plot(x.cpu().detach().numpy(), exact_diffusion.cpu().detach().numpy(), label="true", linewidth = 3,color = 'red')
    plt.plot(x.cpu().detach().numpy(), pred_diffusion.cpu().detach().numpy(), label="predicted", linewidth = 3,color = 'blue')
    plt.xlabel("x")
    plt.ylabel("diffusion")
    plt.ylim(bottom=0, top=2)
    plt.xticks(np.linspace(-intervalradius,intervalradius,3))
    plt.yticks(np.linspace(0,2,3))
    plt.legend(loc = 'upper right')
    img = wandb.Image(plt)
    try:
        wandb.log({'Image_diffusion': img})
    except:
        pass
    plt.savefig("save_files/Image_diffusion.png")
    plt.close()


def plot_2d(drift_fun,drift_net,diffusion_fun,diffusion_net,device,intervalradius,model):
    interval=[-intervalradius,intervalradius]
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
            exact_drift[i,j,:]= drift_fun(data[i,j,:].unsqueeze(0))[0]
            exact_diffusion[i,j,:,:]=diffusion_fun(data[i,j,:].unsqueeze(0))[0]
            pred_drift[i,j,:]= drift_net(data[i,j,:].unsqueeze(0))[0]
            pred_diffusion[i,j,:]=diffusion_net(data[i,j,:].unsqueeze(0))[0]
    #exact_drift = drift_fun(data)# n,2
    #self.net.load_state_dict(torch.load('1D_drift.pth'))
    #exact_diffusion = diffusion_fun(data) # n,2,2

   
    #pred_drift = drift_net(data)# n,2
    #pred_diffusion = torch.sqrt(diffusion_net(data))# n,2
    zlim = 1 if model == "pdcd" else 3
    plt.figure(figsize = (20,9))
    plt.subplot(1,2,1)
    levels = np.linspace(-zlim,zlim,51)
    plt.contourf(X.detach().cpu().numpy(), Y.detach().cpu().numpy(), exact_drift.detach().cpu().numpy()[...,0],80, cmap='rainbow',levels = levels)
    plt.xticks(np.linspace(-1,1,3))
    plt.yticks(np.linspace(-1,1,3))
    plt.title("true_drift_X")
    cbar = plt.colorbar() 
    cbar.set_ticks(np.linspace(-zlim,zlim,5))
    #plt.savefig("results/exact1.png")
    #plt.show()
    #plt.close()
    plt.subplot(1,2,2)
    plt.contourf(X.detach().cpu().numpy(), Y.detach().cpu().numpy(), exact_drift.detach().cpu().numpy()[...,1],80, cmap='rainbow',levels = levels)
    plt.xticks(np.linspace(-1,1,3))
    plt.yticks(np.linspace(-1,1,3))
    plt.title("true_drift_Y")
    cbar = plt.colorbar() 
    cbar.set_ticks(np.linspace(-zlim,zlim,5))  
    img = wandb.Image(plt)
    try:
        wandb.log({'Image_exact': img})
    except:
        pass
    plt.close()


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
    
    '''
    levelsdif = np.linspace(-2,2,51)
    plt.subplot(4,2,5)
    plt.contourf(X.detach().cpu().numpy(), Y.detach().cpu().numpy(), exact_diffusion.detach().cpu().numpy()[...,0,0],80, cmap='rainbow',levels=levelsdif)
    plt.title("exact_diffusion_X")
    plt.colorbar()
    #plt.savefig("results/exact1.png")
    #plt.show()
    #plt.close()
    plt.subplot(4,2,6)
    plt.contourf(X.detach().cpu().numpy(), Y.detach().cpu().numpy(), exact_diffusion.detach().cpu().numpy()[...,1,1],80, cmap='rainbow',levels=levelsdif)
    plt.title("exact_diffusion_Y")
    plt.colorbar()
    plt.subplot(4,2,7)
    plt.contourf(X.detach().cpu().numpy(), Y.detach().cpu().numpy(), pred_diffusion.detach().cpu().numpy()[...,0],80, cmap='rainbow',levels=levelsdif)
    plt.title("pred_diffusion_X")
    plt.colorbar()
    plt.subplot(4,2,8)
    plt.contourf(X.detach().cpu().numpy(), Y.detach().cpu().numpy(), pred_diffusion.detach().cpu().numpy()[...,1],80, cmap='rainbow',levels=levelsdif)
    plt.title("pred_diffusion_Y")
    plt.colorbar()
    '''

    img = wandb.Image(plt)
    try:
        wandb.log({'Image_pred': img})
    except:
        pass
    plt.close()