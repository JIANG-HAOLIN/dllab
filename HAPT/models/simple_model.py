import torch
import torch.nn as nn
from inputpipeline import datasets


class model(nn.Module):
    def __init__(self,batchsize,device,hidden_size,num_layers,bidirectional,window_size):
        super(model,self).__init__()
        self.rnn = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.batch_size = batchsize
        self.coeffienct = 1 if bidirectional == False else 2
        self.hidden_size = hidden_size
        self.window_size = window_size
        #输入[Batchsize,windows length, 6]
        self.rnn.append(nn.LSTM(input_size=6,hidden_size=hidden_size,num_layers=num_layers,bidirectional=bidirectional,dropout=0.2))
        ##输出[B,250,1或2xhidden_size]
        self.h0 = torch.randn(self.coeffienct*num_layers,self.batch_size,hidden_size).to(device)##(num_layers*num_direction,batch,hidden_size)
        self.c0 = torch.randn(self.coeffienct*num_layers,self.batch_size,hidden_size).to(device)

        # self.rnn.append(nn.LSTM(input_size=2*32, hidden_size=16, num_layers=2, bidirectional=True, batch_first=True,dropout=0.2))
        # ##[B,250,2xhidden_size]
        # self.h1 = torch.randn(2 * 2, self.batch_size, 16)
        # self.c1 = torch.randn(2 * 2, self.batch_size, 16)
        self.linears.append(nn.Flatten())
        self.linears.append(nn.Linear(self.coeffienct*hidden_size*self.window_size,self.coeffienct*hidden_size))
        self.linears.append(nn.ReLU())
        self.linears.append(nn.Linear(self.coeffienct*hidden_size,self.coeffienct*hidden_size))
        self.linears.append(nn.Tanh())
        self.linears.append(nn.Linear(self.coeffienct*hidden_size,12))
        self.linears.append(nn.Softmax(dim=1))
        ##output [B,12]
    def forward(self,input):
        input = input.permute(1,0,2)
        output,_ = self.rnn[0](input,(self.h0,self.c0))
        output = output.permute(1,0,2)

        # output = output.permute(0,1,2)
        # output = output[:,-1,:]##last time stamp output



        for linear in self.linears:
            output = linear(output)

        return output


#test model
# mdl = model(batchsize=32)
# output = mdl(torch.randn(32,250,6))
# print(output)


