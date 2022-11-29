import torch
import torch.nn as nn
from tqdm import tqdm
from models.architectures import MyModel, efficient_model, efficient_model_reg
from input_pipeline.datasets import *
from torch.nn import BCELoss
import torch.optim as optim
import matplotlib.pyplot as plt
from evaluation.metrics import compute_matrix

reg = True

device = "cuda"

if not reg:
    loss_fc = BCELoss()
    mdl = efficient_model.to(device)
    train_loader = train_loader
else:
    loss_fc = nn.MSELoss()
    mdl = efficient_model_reg.to(device)
    train_loader = train_loader_reg
optimizer = optim.Adam(mdl.parameters(), lr=3e-5, weight_decay=5e-4)

store = []
store_acu = []
store_tp = []
store_tn = []
store_fp = []
store_fn = []
cur = []
epoch = 1000

for epc in range(epoch):
    for idx, i in tqdm(enumerate(train_loader)):
        cur_iter = epc*413 + idx + 1
        img = i[0]
        label = i[1]
        img = img.to(device)
        y = mdl(img)
        y = y.squeeze(1)
        label = label.to(torch.float).to(device)

        loss = loss_fc(y, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if cur_iter % 10 == 0:
            cur.append(cur_iter)
            mdl.eval()
            loss_val = 0
            acu, tp, tn, fp, fn = 0, 0, 0, 0, 0
            for idx2, j in tqdm(enumerate(test_loader)):
                with torch.no_grad():
                    mdl.eval()
                    img = j[0]
                    label = j[1]
                    img = img.to(device)
                    y = mdl(img)
                    y = y.squeeze(1)
                    if reg:
                        y[(y >= 0.7)] = 1
                        y[(y < 0.7)] = 0
                    label = label.to(torch.float).to(device)
                    loss_val = (loss_val*idx2+loss_fc(y, label))/(idx2+1)
                    _, accuracy = compute_matrix(y.detach().cpu().numpy(), label.detach().cpu().numpy())
                    tp, tn, fp, fn = (tp * idx2 + _[0]) / (idx2 + 1), (tn * idx2 + _[1]) / (idx2 + 1), (fp * idx2 + _[2])/(idx2 + 1), (fn * idx2 + _[3]) / (idx2 + 1)
                    acu = (acu * idx2 + accuracy) / (idx2 + 1)

            mdl.train()

            store.append(loss_val.item())
            store_acu.append(acu)
            store_tp.append(tp)
            store_tn.append(tn)
            store_fp.append(fp)
            store_fn.append(fn)

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













