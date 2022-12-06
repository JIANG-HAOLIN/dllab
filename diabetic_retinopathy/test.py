import torch
from tqdm import tqdm
from models.architectures import *
from input_pipeline.datasets import *
from torch.nn import BCELoss
import torch.optim as optim
import matplotlib.pyplot as plt
from evaluation.metrics import compute_matrix


import config


import cv2
import argparse
from tqdm import tqdm
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from input_pipeline.datasets import *
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit



opt = config.read_arguments()
opt.regression = False
device = opt.device
reg = opt.regression
root_path = opt.root_path
print(torch.cuda.is_available())


store = []
store_acu = []
cur = []
epoch = 500
acu_best = 0
threshold = 0.8



mdl = get_model(reg).to(device)
mdl.load_state_dict(torch.load(root_path+'/'+opt.out_name + '.pth',map_location=torch.device('cpu')))
mdl.eval()

loss_fc = BCELoss() if not reg else nn.MSELoss()
loss_val = 0
acu = 0

test_loader = get_kaggle_test_loader(root_path=root_path,which='kaggle',batch_size=12,wanted_size=728,reg=False)
for idx2, j in tqdm(enumerate(test_loader)):
    with torch.no_grad():
        mdl.eval()
        img = j[0]
        label = j[1]
        img = img.to(device)
        y = mdl(img)
        y = y.squeeze(1)

        if reg:
            y[(y >= threshold)] = 1
            y[(y < threshold)] = 0

        label = label.to(torch.float).to(device)
        loss_val = (loss_val * idx2 + loss_fc(y, label)) / (idx2 + 1)
        _, accuracy = compute_matrix(y.detach().cpu().numpy(), label.detach().cpu().numpy())
        acu = (acu * idx2 + accuracy) / (idx2 + 1)
        if acu > acu_best:
            acu_best = acu
            best_state_dict = mdl.state_dict()
print(acu)
print(loss_val.item())
# store.append(loss_val.item())
# store_acu.append(acu)
# fig = plt.figure()
# plt.plot(cur, store)  # plot example
# fig.savefig('loss_%s_%s.png' % (opt.wanted_size, opt.out_name))
#
# fig2 = plt.figure()
# plt.plot(cur, store_acu)  # plot example
# fig2.savefig('accuracy_%s_%s.png' % (opt.wanted_size, opt.out_name))

