import torch
from tqdm import tqdm
from models.architectures import *
from input_pipeline.datasets_new import *
from torch.nn import BCELoss
import torch.optim as optim
import matplotlib.pyplot as plt
from evaluation.metrics import compute_matrix

import config

opt = config.read_arguments()
device = opt.device
reg = opt.regression
root_path = opt.root_path
print(torch.cuda.is_available())


loss_fc = BCELoss() if not reg else nn.MSELoss()
mdl = get_model(reg).to(device)
optimizer = optim.Adam(mdl.parameters(), lr=3e-5, weight_decay=5e-4)

store = []
store_acu = []
cur = []
epoch = 500
acu_best = 0
threshold = 0.8

train_loader,test_loader = get_both_loader(
                                        root_path=opt.root_path,
                                        which='kaggle',
                                        batch_size=opt.batch_size,
                                        wanted_size=opt.wanted_size,
                                        reg=reg
                                      )
for epc in range(epoch):
    print("the %d - th epoch :" %epc)
    for idx, i in tqdm(enumerate(train_loader)):
        cur_iter = epc*413 + idx + 1
        img = i[0]
        label = i[1]
        img = img.to(device)

        # img = stack(img)

        y = mdl(img)
        y = y.squeeze(1)
        print(y)

        label = label.to(torch.float).to(device)
        # print(label)

        # label = torch.cat((label,label,label,label),0)

        loss = loss_fc(y, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if cur_iter % 10 == 0:
            cur.append(cur_iter)
            mdl.eval()
            loss_val = 0
            acu = 0
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
                    loss_val = (loss_val*idx2+loss_fc(y, label))/(idx2+1)
                    _, accuracy = compute_matrix(y.detach().cpu().numpy(), label.detach().cpu().numpy())
                    acu = (acu * idx2 + accuracy) / (idx2 + 1)
                    if acu > acu_best:
                        acu_best = acu
                        best_state_dict = mdl.state_dict()



            mdl.train()




            store.append(loss_val.item())
            store_acu.append(acu)
            fig = plt.figure()
            plt.plot(cur, store)  # plot example
            fig.savefig('loss_%s_%s.png' %(opt.wanted_size,opt.out_name))

            fig2 = plt.figure()
            plt.plot(cur, store_acu)  # plot example
            fig2.savefig('accuracy_%s_%s.png' %(opt.wanted_size,opt.out_name))

torch.save(best_state_dict, root_path+'/'+opt.out_name + '.pth')












