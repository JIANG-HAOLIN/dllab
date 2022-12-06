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

reg = False
writer = SummaryWriter("logs")
device = "cuda"


if not reg:
    loss_fc = nn.CrossEntropyLoss()
    mdl = get_efficient_model().to(device)
    train_loader = train_loader
else:
    loss_fc = nn.MSELoss()
    mdl = get_efficient_model(reg=True).to(device)
    train_loader = train_loader_reg
optimizer = optim.Adam(mdl.parameters(), lr=3e-5, weight_decay=5e-3)

def load_best(model):
    model2 = torch.load("./best_epoch.pth")
    model.load_state_dict(model2)



store = []
store_acu = []
store_tp = []
store_tn = []
store_fp = []
store_fn = []
cur = []
epoch = 1000
best_acu = 0

for epc in range(epoch):
    for idx, i in tqdm(enumerate(train_loader)):
        cur_iter = epc*413 + idx + 1
        img = i[0]
        label = i[1]
        img = img.to(device)
        y = mdl(img)
        y = y.squeeze(1)
        label = label.to(torch.long).to(device)
        # label = label.to(device)
        loss = loss_fc(y, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if cur_iter % 10 == 0:
            writer.add_scalar("train loss", loss.item(), cur_iter)
            cur.append(cur_iter)
            mdl.eval()
            loss_val = 0
            acu, tp, tn, fp, fn = 0, 0, 0, 0, 0
            mdl.eval()

            for idx2, j in tqdm(enumerate(test_loader)):
                with torch.no_grad():
                    img = j[0]
                    label = j[1]
                    img = img.to(device)
                    y = mdl(img)
                    y = y.squeeze(1)
                    if reg:
                        y[(y >= 0.4)] = 1
                        y[(y < 0.4)] = 0
                    label = label.to(torch.long).to(device)
                    loss_val = (loss_val*idx2+loss_fc(y, label))/(idx2+1)
                    _, accuracy = compute_matrix_CE(y.detach().cpu().numpy(), label.detach().cpu().numpy())
                    tp, tn, fp, fn = (tp * idx2 + _[0]) / (idx2 + 1), (tn * idx2 + _[1]) / (idx2 + 1), (fp * idx2 + _[2])/(idx2 + 1), (fn * idx2 + _[3]) / (idx2 + 1)
                    acu = (acu * idx2 + accuracy) / (idx2 + 1)

            mdl.train()
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

writer.close()











