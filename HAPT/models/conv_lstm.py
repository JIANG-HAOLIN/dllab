import torch
import torch.nn as nn
from inputpipeline import datasets
from models.transformer import Encoder
from einops.layers.torch import Rearrange



class Conv_lstm(nn.Module):
    def __init__(self,device='cpu',num_lstm_layers=2,hidden_size=96,batch_size=32,how=None):
        super(Conv_lstm, self).__init__()
        print('structure:Conv_lstm'+'\nbidirectional='+ f'{True}, '+'batchsize='+f'{batch_size}, '+'hidden_size='+f'{hidden_size}')
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=6,out_channels=12,kernel_size=3,stride=1,padding='same',),
            nn.BatchNorm1d(12),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=12,out_channels=24,kernel_size=3,stride=1,padding='same',),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
## [ 32 , 24 , 62 ]
            nn.Conv1d(in_channels=24, out_channels=48, kernel_size=3, stride=1, padding='same', ),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

        )
        self.rnn = nn.ModuleList()
        self.rnn.append(nn.LSTM(input_size=48,
                    hidden_size=96,
                    num_layers=num_lstm_layers,
                    bidirectional=True,
                    dropout=0.2))
        self.h0 = torch.randn( 2 * num_lstm_layers, batch_size , hidden_size).to(device)
        ##(num_layers*num_direction,batch,hidden_size)
        self.c0 = torch.randn( 2 * num_lstm_layers, batch_size, hidden_size).to(device)
        self.lstm_act = nn.Tanh()

        self.linears = nn.Sequential(
            nn.Flatten(),
            nn.Linear( 2 * hidden_size * 31, 2 * hidden_size),
            nn.BatchNorm1d(num_features= 2 * hidden_size),
            nn.ReLU(),
            nn.Linear( 2 * hidden_size, 2 * hidden_size),
            nn.BatchNorm1d(num_features= 2 * hidden_size),
            nn.Tanh(),
            nn.Linear( 2 * hidden_size, 12),
            nn.BatchNorm1d(num_features=12),
            nn.Softmax(dim=1)

        )

    def forward(self, input):
        input = input.permute(0,2,1)
        output = self.conv_layers(input)
        # print(output.shape)
        output = output.permute(2, 0, 1)
        # print(output.shape)

        output, _ = self.rnn[0](output, (self.h0,self.c0))
        output = output.permute(1, 0, 2)
        # output = output.permute(0,1,2)
        # output = output[:,-1,:]##last time stamp output

        output = self.linears(output)
        # print(output.shape)
        return output

# mdl = Conv_lstm()
# input = torch.randn(32,250,6)
# output = mdl(input)


class Conv_lstm_realworld(nn.Module):
    def __init__(self,device='cpu',num_lstm_layers=2,hidden_size=96,batch_size=32):
        super(Conv_lstm_realworld, self).__init__()
        print('structure:Conv_lstm_realworld'+'\nbidirectional='+ f'{True}, '+'batchsize='+f'{batch_size}, '+'hidden_size='+f'{hidden_size}')
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=12, kernel_size=3, stride=1, padding='same', ),
            nn.BatchNorm1d(12),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding='same', ),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            ## [ 32 , 24 , 62 ]
            nn.Conv1d(in_channels=24, out_channels=48, kernel_size=3, stride=1, padding='same', ),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

        )
        self.rnn = nn.ModuleList()
        self.rnn.append(nn.LSTM(input_size=48,
                                hidden_size=96,
                                num_layers=num_lstm_layers,
                                bidirectional=True,
                                dropout=0.2))
        self.h0 = torch.randn(2 * num_lstm_layers, batch_size, hidden_size).to(device)
        ##(num_layers*num_direction,batch,hidden_size)
        self.c0 = torch.randn(2 * num_lstm_layers, batch_size, hidden_size).to(device)
        self.lstm_act = nn.Tanh()

        self.linears = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * hidden_size * 31, 2 * hidden_size),
            nn.BatchNorm1d(num_features=2 * hidden_size),
            nn.ReLU(),
            nn.Linear(2 * hidden_size, 2 * hidden_size),
            nn.BatchNorm1d(num_features=2 * hidden_size),
            nn.Tanh(),
            nn.Linear(2 * hidden_size, 8),
            nn.BatchNorm1d(num_features=8),
            nn.Softmax(dim=1)

        )

    def forward(self, input):
        input = input.permute(0,2,1)
        output = self.conv_layers(input)
        # print(output.shape)
        output = output.permute(2, 0, 1)
        # print(output.shape)

        output, _ = self.rnn[0](output, (self.h0,self.c0))
        output = output.permute(1, 0, 2)
        # output = output.permute(0,1,2)
        # output = output[:,-1,:]##last time stamp output

        output = self.linears(output)
        # print(output.shape)
        return output


# mdl = Conv_lstm_realworld()
# input = torch.randn(32,250,6)
# output = mdl(input)
# print(output.shape)

class cnn_lstm_cnn_HAPT(nn.Module):
    def __init__(self,batchsize,device,hidden_size,num_layers,bidirectional,window_size,how='s2l'):
        super(cnn_lstm_cnn_HAPT,self).__init__()
        print('structure:CNN_LSTM_CNN_HAPT'+'\nbidirectional='+ f'{bidirectional}, '+'batchsize='+f'{batchsize}, '+'hidden_size='+f'{hidden_size}')
        self.rnn = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.batch_size = batchsize
        self.coeffienct = 1 if bidirectional == False else 2
        self.hidden_size = hidden_size
        self.window_size = window_size
        #input [Batchsize,windows length, 6]
        self.conv_layers = nn.Sequential(
            Rearrange('b l c -> b c l'),
            nn.Conv1d(in_channels=6, out_channels=12, kernel_size=15, stride=1, padding='same', ),
            nn.BatchNorm1d(12),
            nn.ReLU(),
            Rearrange('b c l -> b l c'),
        )
        self.rnn.append(nn.LSTM(input_size=12,hidden_size=hidden_size,num_layers=num_layers,bidirectional=bidirectional,dropout=0.2))
        ##[B,250,1或2xhidden_size]
        self.h0 = torch.randn(self.coeffienct*num_layers,self.batch_size,hidden_size).to(device)##(num_layers*num_direction,batch,hidden_size)
        self.c0 = torch.randn(self.coeffienct*num_layers,self.batch_size,hidden_size).to(device)

        # self.rnn.append(nn.LSTM(input_size=2*32, hidden_size=16, num_layers=2, bidirectional=True, batch_first=True,dropout=0.2))
        # ##[B,250,2xhidden_size]
        # self.h1 = torch.randn(2 * 2, self.batch_size, 16)
        # self.c1 = torch.randn(2 * 2, self.batch_size, 16)
        ##output [B,250,12]
        self.linears.append(Rearrange('b l c -> b c l'))
        self.linears.append(nn.BatchNorm1d(num_features=self.coeffienct*hidden_size)),
        self.linears.append(nn.Tanh())
        self.linears.append(nn.Dropout(0.2))
        self.linears.append(nn.Conv1d(in_channels=self.coeffienct*hidden_size,out_channels=self.coeffienct*hidden_size,kernel_size=25,padding='same'))
        self.linears.append(nn.BatchNorm1d(num_features=self.coeffienct*hidden_size)),
        self.linears.append(nn.ReLU())
        self.linears.append(nn.Dropout(0.2))
        self.linears.append(nn.Conv1d(in_channels=self.coeffienct*hidden_size,out_channels=12,kernel_size=25,padding='same'))
        self.linears.append(nn.BatchNorm1d(num_features=12)),
        self.linears.append(Rearrange('b c l -> b l c'))
        self.linears.append(nn.Softmax(dim=-1))
    def forward(self,input):
        output = self.conv_layers(input)
        input = output.permute(1,0,2)
        output,_ = self.rnn[0](input,(self.h0,self.c0))
        output = output.permute(1,0,2)

        # output = output.permute(0,1,2)
        # output = output[:,-1,:]##last time stamp output

        for linear in self.linears:
            output = linear(output)

        return output