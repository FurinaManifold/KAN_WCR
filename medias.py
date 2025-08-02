import torch
import wandb
import warnings
import argparse
import yaml
from pytorch_lightning.loggers import WandbLogger
from model.net import net_select
from utils.functions import model_select
from utils.functions import diffusion_knownnet
from utils.plot import plot_error
import time

def diffusion_Chen(x):
    return 1.0138*torch.ones_like(x)

def drift_Chen(x):
    return 0.0225*torch.ones_like(x)+0.9638*x-0.0010*x**2-1.0035*x**3

def diffusion_Chen2(x):
    return torch.tensor([0.9863,0.9731])

def drift_Chen2(x):
    f1 = 0.0022+0.9966*x[0]-0.0049*x[0]**2-0.9755*x[0]**3
    f2 = -0.0264+0.9948*x[1]+0.0162*x[1]**2-0.9526*x[1]**3
    return torch.tensor([f1,f2])

def drawChen2(file):
    with open(file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    wandb_logger = WandbLogger()
    wandb.init(project='WCR_integrated_graphs4',name = 'Chend')
    wandb_logger.experiment.config.update(config,'allow_val_change=True')
    drift,diffusion = model_select('pdcd',0)
    net_drift = drift_Chen
    net_diffusion = diffusion_Chen
    plot_error(drift,net_drift,diffusion,net_diffusion,'cuda','2d',config['data'])

def drawChen(file):
    with open(file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    wandb_logger = WandbLogger()
    wandb.init(project='WCR_integrated_graphs2',name = 'Chenb')
    wandb_logger.experiment.config.update(config,'allow_val_change=True')
    drift,diffusion = model_select('pdcd',0)
    net_drift = drift_Chen
    net_diffusion = diffusion_Chen
    plot_error(drift,net_drift,diffusion,net_diffusion,'cuda','1d',config['data'])

def drawfigures(file):
    print(f"Handling {file}\n")
    with open(file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    wandb_logger = WandbLogger()
    wandb.init(project='WCR_integrated_graphs3',name = config['Project']['name'])
    wandb_logger.experiment.config.update(config,'allow_val_change=True')
    net_drift , net_diffusion = net_select(config['NN'])
    if config['train']['known_diffusion']:
        net_diffusion = diffusion_knownnet
    else: 
        net_diffusion = torch.load(f"./modelckpts/{config['Project']['name']}_diffusion.ckpt", map_location = 'cuda')
    net_drift = torch.load(f"./modelckpts/{config['Project']['name']}_drift.ckpt",map_location = 'cuda')
    # print('--------------waiting----------------')
    # time.sleep(10)
    drift,diffusion = model_select(config['data']['model'],config['data']['frequency'])
    plot_error(drift,net_drift,diffusion,net_diffusion,'cuda',config['train']['plot_type'],config['data'])


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser('Training of the Architectures', add_help=True)
    parser.add_argument('-c','--config_file', type=str)
    args = parser.parse_args()
    drawfigures(args.config_file)
    # file = 'configs/part_snapshots/config_2D_poly_KAN_3_2_30_constdif.yaml'
    # drawChen2(file)