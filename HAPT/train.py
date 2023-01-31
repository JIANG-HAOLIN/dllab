import wandb
import math
import torch
from models.multi_models import model_HAPT,model_HAR,model_gru_HAPT,model_transformer_HAPT
from inputpipeline.datasets import get_dataloader
from inputpipeline.preprocess_input import preprocess_input
from inputpipeline.HAR_Dataset import get_dataloader_HAR
from Trainer import Trainer


import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from evaluation.metrics import compute_accuracy
import config


opt = config.read_arguments()
batch_size = opt.batch_size
root_path =opt.root_path
shift_length = opt.shift_length
window_size = opt.window_size
device = opt.device
hidden_size = opt.hidden_size
num_layers = opt.num_layers
lr = opt.lr
bidirectional = opt.bidirectional
dataset = opt.dataset
structure = opt.structure
out_name = opt.dataset+'_'+opt.structure+'_'+opt.out_name+'b'+str(batch_size)+'hidden'+str(hidden_size)+'layer'+str(num_layers)
epoch = opt.epoch

loss_computer = torch.nn.CrossEntropyLoss()
if dataset == 'HAPT':
    train_loader = get_dataloader(mode='train',Window_shift=125,Window_length=250,
                                  batch_size=batch_size,shuffle=True,root_path='./RawData/')
    validation_loader = get_dataloader(mode='validation',Window_shift=125,Window_length=250,
                                       batch_size=batch_size,shuffle=True,root_path='./RawData/')
    if structure == 'lstm':
        mdl = model_HAPT(batchsize=batch_size, device=device, hidden_size=hidden_size,
                         num_layers=num_layers, bidirectional=bidirectional, window_size=window_size).to(device)
    elif structure == 'gru':
        mdl = model_gru_HAPT(batchsize=batch_size, device=device, hidden_size=hidden_size,
                         num_layers=num_layers, bidirectional=bidirectional, window_size=window_size).to(device)
    elif structure == 'transformer':
        mdl = model_transformer_HAPT(batchsize=batch_size)
elif dataset == 'HAR':
    train_loader = get_dataloader_HAR(mode='train',Window_shift=125,Window_length=250,batch_size=batch_size,
                                      shuffle=True,root_path='./realworld2016_dataset/')
    validation_loader = get_dataloader_HAR(mode='validation',Window_shift=125,Window_length=250,batch_size=batch_size,
                                           shuffle=True,root_path='./realworld2016_dataset/')
    if structure == 'lstm':
        mdl = model_HAR(batchsize=batch_size, device=device, hidden_size=hidden_size,
                         num_layers=num_layers, bidirectional=bidirectional, window_size=window_size).to(device)
    elif structure == 'transformer':
        mdl = model_transformer_HAPT(batchsize=batch_size)


def train(config=None):
    with wandb.init(config=None):
        config = wandb.config
        print(config)
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        Trainer(mdl=mdl,
                loss_computer=loss_computer,
                optimizer=torch.optim.Adam(mdl.parameters(), lr=config.learning_rate, weight_decay=5e-3),
                epoch=epoch,
                learning_rate=2e-4,
                train_loader=train_loader,
                validation_loader=validation_loader,
                out_name=out_name,
                device=device,
                batch_size=batch_size,
                )

# 超参数搜索方法，可以选择：grid random bayes
sweep_config = {
    'method': 'bayes',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    }

# 参数范围
parameters_dict = {
    # 'num_filter':{
    #     'values':[3, 4, 5]
    #     },
    # 'optimizer': {
    #     'values': ['adam', 'sgd']
    #     },
    # 'dropout': {
    #       'values': [0.3, 0.4, 0.5]
    #     },
    'learning_rate': {
        # a flat distribution between 0 and 0.1
        'values':[1e-3,1e-4,1e-5]
      },
    'epoch': {
        'values': [5, 10, 15]
    },
    'batch_size': {
        # integers between 32 and 256
        # with evenly-distributed logarithms
        # 'distribution': 'q_log_uniform',
        # 'q': 1,
        # 'min': math.log(32),
        # 'max': math.log(256),
        'values':[32,64]
      },
    'hidden_size': {
        'values': [12, 18, 24]
    },
    'num_layers': {
        'values': [1, 2, 3]
    },
    'windows_size': {
        'values': [150,200,250]
    },

    }

sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")
# wandb.init()

if __name__ == '__main__':
    wandb.agent(sweep_id, train, count=50)






