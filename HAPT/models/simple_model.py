import torch
import torch.nn as nn
from HAPT.inputpipeline import datasets


class model(nn.Module):
    def __init__(self):
        super(model,self).__init__()
        self.rnn = nn.ModuleList()
        self.linears = nn.ModuleList()
        #[3,250,6]
        self.rnn.append(nn.LSTM(input_size=6,hidden_size=12,num_layers=2,bidirectional=True,batch_first=True))
        ##[3,250,2x12]
        self.h0 = torch.randn(2*2,3,12)
        self.c0 = torch.randn(2*2,3,12)
        self.rnn.append(nn.LSTM(input_size=2*12, hidden_size=12, num_layers=2, bidirectional=True, batch_first=True))
        ##[3,250,2x12]
        self.h1 = torch.randn(2 * 2, 3, 12)
        self.c1 = torch.randn(2 * 2, 3, 12)

        self.linears.append(nn.Linear(2*12,1))
        self.linears.append(nn.Dropout(0.3))
        self.linears.append(nn.Flatten())
        self.linears.append(nn.Linear(250,12))
        self.linears.append(nn.Softmax(dim=1))
        ##output [3,12]
    def forward(self,input):
        output,_ = self.rnn[0](input,(self.h0,self.c0))
        output,_ = self.rnn[1](output,(self.h1,self.c1))

        for linear in self.linears:
            output = linear(output)

        return output


