import torch
from tqdm import tqdm
from models.architectures import MyModel
from input_pipeline.datasets import *
from torch.nn import BCELoss
import torch.optim as optim
import matplotlib.pyplot as plt

loss_fc = BCELoss()
device = "cuda"


mdl = MyModel(3).to(device)
optimizer = optim.SGD(lr=0.00001, params=mdl.parameters(), weight_decay=20)

store = []
epoch = 100

for epc in range(epoch):
    for idx, i in tqdm(enumerate(train_loader)):
        img = i[0]
        label = i[1]
        img = img.to(device)
        y = mdl(img)
        print(y.shape)
        label = label.to(torch.float).to(device)

        loss = loss_fc(y, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx // 1 == 0:
            loss_val = 0
            for idx2, j in tqdm(enumerate(test_loader)):
                with torch.no_grad():
                    mdl.eval()
                    img = j[0]
                    label = j[1]
                    img = img.to(device)
                    y = mdl(img)
                    label = label.to(torch.float).to(device)
                    loss_val = (loss_val*(idx2 + 1)+loss_fc(y, label))/(idx2+2)

            store.append(loss_val.item())







fig = plt.figure()
plt.plot(store)  # plot example
fig.savefig('temp.png')



