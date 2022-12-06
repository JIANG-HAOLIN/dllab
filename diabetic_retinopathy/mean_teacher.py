import torch
import torch.nn as nn
from tqdm import tqdm
from models.architectures import get_efficient_model
from input_pipeline.datasets import *
from torch.nn import BCELoss
import torch.optim as optim
import matplotlib.pyplot as plt
from evaluation.metrics import compute_matrix, compute_matrix_CE
from torch.utils.tensorboard import SummaryWriter
from utils import losses, ramps

reg = False
ema = True
writer = SummaryWriter("logs")
device = "cuda"


if not reg:
    loss_fc = nn.CrossEntropyLoss()
    mdl = get_efficient_model(pretrained=True).to(device)
    train_loader = train_loader
else:
    loss_fc = nn.MSELoss()
    mdl = get_efficient_model(ema=False, reg=True).to(device)
    train_loader = train_loader_reg

if ema:
    ema_mdl = get_efficient_model(ema=True, reg=False, pretrained=True).to(device)

optimizer = optim.Adam(mdl.parameters(), lr=3e-5, weight_decay=5e-3)
# optimizer = torch.optim.SGD(mdl.parameters(), 3e-5,
#                             momentum=0.9,
#                             weight_decay=2e-4,
#                             nesterov=True)

def load_best(model):
    model2 = torch.load("./best_epoch.pth")
    model.load_state_dict(model2)

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 100.0 * ramps.sigmoid_rampup(epoch, 5)

def update_ema_variables(model, ema_model, global_step, alpha=0.98):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

store = []
store_acu = []
store_tp = []
store_tn = []
store_fp = []
store_fn = []
cur = []
round = 100000
best_acu = 0

it_labeled = iter(train_loader)
it_unlabeled = iter(kaggle_loader)
residual_logit_criterion = losses.symmetric_mse_loss
consistency_criterion = losses.softmax_mse_loss
epoch = 1
for cur_iter, rd in enumerate(range(round)):
    try:
        img1, label1 = next(it_labeled)
    except StopIteration:
        it_labeled = iter(train_loader)
        img1, label1 = next(it_labeled)
    minibatch_size = img1.shape[0]
    img1 = img1.to(device)
    label1 = label1.to(torch.long).to(device)
    class_logit, cons_logit = mdl(img1)
    class_loss = loss_fc(class_logit, label1) / minibatch_size
    ema_logit, _ = ema_mdl(img1)
    ema_logit.detach()
    res_loss = 0.01 * residual_logit_criterion(class_logit, cons_logit) / minibatch_size
    consistency_weight = get_current_consistency_weight(epoch)
    consistency_loss = consistency_weight * consistency_criterion(cons_logit, ema_logit) / minibatch_size
    loss = class_loss + consistency_loss/2 + res_loss/2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    try:
        img2, _ = next(it_unlabeled)
    except StopIteration:
        it_unlabeled = iter(kaggle_loader)
        img2, _ = next(it_unlabeled)
        epoch += 1
    minibatch_size = img2.shape[0]
    img2 = img2.to(device)
    class_logit, cons_logit = mdl(img2)
    ema_logit, _ = ema_mdl(img2)
    ema_logit.detach()
    res_loss = 0.01 * residual_logit_criterion(class_logit, cons_logit) / minibatch_size
    consistency_loss = consistency_weight * consistency_criterion(cons_logit, ema_logit) / minibatch_size
    loss = consistency_loss/2 + res_loss/2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    update_ema_variables(mdl, ema_mdl, cur_iter)

    if cur_iter % 10 == 0:
        writer.add_scalar("train loss", loss.item(), cur_iter)
        cur.append(cur_iter)
        mdl.eval()
        loss_val = 0
        acu, tp, tn, fp, fn = 0, 0, 0, 0, 0
        mdl.eval()
        ema_mdl.eval()

        for idx2, j in tqdm(enumerate(test_loader)):
            with torch.no_grad():
                img = j[0]
                label = j[1]
                img = img.to(device)
                y = ema_mdl(img)[0]
                # y = y.squeeze(1)
                if reg:
                    y[(y >= 0.4)] = 1
                    y[(y < 0.4)] = 0
                label = label.to(torch.long).to(device)
                loss_val = (loss_val*idx2+loss_fc(y, label))/(idx2+1)
                _, accuracy = compute_matrix_CE(y.detach().cpu().numpy(), label.detach().cpu().numpy())
                tp, tn, fp, fn = (tp * idx2 + _[0]) / (idx2 + 1), (tn * idx2 + _[1]) / (idx2 + 1), (fp * idx2 + _[2])/(idx2 + 1), (fn * idx2 + _[3]) / (idx2 + 1)
                acu = (acu * idx2 + accuracy) / (idx2 + 1)

        mdl.train()
        ema_mdl.train()

        store.append(loss_val.item())
        store_acu.append(acu)
        store_tp.append(tp)
        store_tn.append(tn)
        store_fp.append(fp)
        store_fn.append(fn)
        writer.add_scalar("test loss", loss_val.item(), cur_iter)
        writer.add_scalar("test accuracy", acu, cur_iter)

        if acu > best_acu:
            best_acu = acu
            torch.save(mdl.state_dict(), "./best_epoch.pth")


        fig = plt.figure()
        plt.plot(cur, store, label="val_loss")  # plot example
        plt.legend()
        fig.savefig('loss.png')

        fig2 = plt.figure()
        plt.plot(cur, store_acu, label="accuracy")  # plot example
        plt.legend()
        fig2.savefig('accuracy.png')

        fig3 = plt.figure()
        plt.plot(cur, store_tp, label="TP")  # plot example
        plt.plot(cur, store_tn, label="TN")  # plot example
        plt.plot(cur, store_fp, label="FP")  # plot example
        plt.plot(cur, store_fn, label="FN")  # plot example
        plt.legend(loc='upper left')
        fig3.savefig('Confusion_Matrix.png')
        plt.cla()

writer.close()











