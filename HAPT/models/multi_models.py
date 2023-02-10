import torch
import torch.nn as nn
from HAPT.inputpipeline import datasets
from HAPT.models.transformer import Encoder



class model_HAPT(nn.Module):
    def __init__(self,batchsize,device,hidden_size,num_layers,bidirectional,window_size,how='s2l'):
        super(model_HAPT,self).__init__()
        print('structure:LSTM_HAPT'+'\nbidirectional='+ f'{bidirectional}, '+'batchsize='+f'{batchsize}, '+'hidden_size='+f'{hidden_size}')
        self.rnn = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.batch_size = batchsize
        self.coeffienct = 1 if bidirectional == False else 2
        self.hidden_size = hidden_size
        self.window_size = window_size
        #input [Batchsize,windows length, 6]
        self.rnn.append(nn.LSTM(input_size=6,hidden_size=hidden_size,num_layers=num_layers,bidirectional=bidirectional,dropout=0.2))
        ##输出[B,250,1或2xhidden_size]
        self.h0 = torch.randn(self.coeffienct*num_layers,self.batch_size,hidden_size).to(device)##(num_layers*num_direction,batch,hidden_size)
        self.c0 = torch.randn(self.coeffienct*num_layers,self.batch_size,hidden_size).to(device)

        # self.rnn.append(nn.LSTM(input_size=2*32, hidden_size=16, num_layers=2, bidirectional=True, batch_first=True,dropout=0.2))
        # ##[B,250,2xhidden_size]
        # self.h1 = torch.randn(2 * 2, self.batch_size, 16)
        # self.c1 = torch.randn(2 * 2, self.batch_size, 16)
        if how == 's2l':
            self.linears.append(nn.Flatten())
            self.linears.append(nn.Linear(self.coeffienct*hidden_size*self.window_size,self.coeffienct*hidden_size))
            # self.linears.append(nn.BatchNorm1d(num_features=self.coeffienct*hidden_size)),
            self.linears.append(nn.ReLU())
            self.linears.append(nn.Linear(self.coeffienct*hidden_size,self.coeffienct*hidden_size))
            # self.linears.append(nn.BatchNorm1d(num_features=self.coeffienct*hidden_size)),
            self.linears.append(nn.Tanh())
            self.linears.append(nn.Linear(self.coeffienct*hidden_size,12))
            # self.linears.append(nn.BatchNorm1d(num_features=12)),
            self.linears.append(nn.Softmax(dim=1))
            ####[B,12]
        elif how == 's2s':
            self.linears.append(nn.Flatten())
            self.linears.append(nn.Linear(self.coeffienct*hidden_size*self.window_size,12*self.window_size))
            self.linears.append(nn.BatchNorm1d(num_features=12*self.window_size)),
            self.linears.append(nn.Tanh())
            self.linears.append(nn.Linear(12*self.window_size,12*self.window_size))
            self.linears.append(nn.BatchNorm1d(num_features=12*self.window_size)),
            self.linears.append(nn.Tanh())
            self.linears.append(nn.Unflatten(1,(self.window_size,12)))
            self.linears.append(nn.Softmax(dim=2))
        ##output [B,250,12]
            # self.linears.append(nn.Linear(self.coeffienct * hidden_size, 12))
            # self.linears.append(nn.Softmax(dim=2))
    def forward(self,input):
        input = input.permute(1,0,2)
        output,_ = self.rnn[0](input,(self.h0,self.c0))
        output = output.permute(1,0,2)

        # output = output.permute(0,1,2)
        # output = output[:,-1,:]##last time stamp output

        for linear in self.linears:
            output = linear(output)

        return output


class model_gru_HAPT(nn.Module):
    def __init__(self,batchsize,device,hidden_size,num_layers,bidirectional,window_size):
        super(model_gru_HAPT,self).__init__()
        print('structure:GRU_HAPT'+'\nbidirectional='+ f'{bidirectional}, '+'batchsize='+f'{batchsize}, '+'hidden_size='+f'{hidden_size}')
        self.rnn = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.batch_size = batchsize
        self.coeffienct = 1 if bidirectional == False else 2
        self.hidden_size = hidden_size
        self.window_size = window_size
        #input[Batchsize,windows length, 6]
        self.rnn.append(nn.GRU(input_size=6,hidden_size=hidden_size,num_layers=num_layers,bidirectional=bidirectional,dropout=0.2))
        ##输出[B,250,1或2xhidden_size]
        self.h0 = torch.randn(self.coeffienct*num_layers,self.batch_size,hidden_size).to(device)##(num_layers*num_direction,batch,hidden_size)

        # self.rnn.append(nn.LSTM(input_size=2*32, hidden_size=16, num_layers=2, bidirectional=True, batch_first=True,dropout=0.2))
        # ##[B,250,2xhidden_size]
        # self.h1 = torch.randn(2 * 2, self.batch_size, 16)
        # self.c1 = torch.randn(2 * 2, self.batch_size, 16)
        self.linears.append(nn.Flatten())
        self.linears.append(nn.Linear(self.coeffienct*hidden_size*self.window_size,self.coeffienct*hidden_size))
        self.linears.append(nn.BatchNorm1d(num_features=self.coeffienct*hidden_size)),
        self.linears.append(nn.ReLU())
        self.linears.append(nn.Linear(self.coeffienct*hidden_size,self.coeffienct*hidden_size))
        self.linears.append(nn.BatchNorm1d(num_features=self.coeffienct*hidden_size)),
        self.linears.append(nn.Tanh())
        self.linears.append(nn.Linear(self.coeffienct*hidden_size,12))
        self.linears.append(nn.BatchNorm1d(num_features=12)),
        self.linears.append(nn.Softmax(dim=1))
        ##output [B,12]
    def forward(self,input):
        input = input.permute(1,0,2)
        output,_ = self.rnn[0](input,self.h0)
        output = output.permute(1,0,2)

        # output = output.permute(0,1,2)
        # output = output[:,-1,:]##last time stamp output



        for linear in self.linears:
            output = linear(output)

        return output



class model_transformer_HAPT(nn.Module):
    def __init__(self,batchsize,
                 seq_len=250,
            patch_size=10,
            num_classes=12,
            dim=512,
            depth=6,
            heads=8,
            mlp_dim=1024,
            channels=6,
            dim_head=64):
        super(model_transformer_HAPT,self).__init__()
        print(
            'structure:transformer_HAPT'+
            '\nsequence_length=' + f'{seq_len}, ' +
            '\npatch_size='+f'{patch_size}, '+
              '\nbatchsize='+f'{batchsize}, '+
              '\ndim='+f'{dim}'+
            '\ndepth=' + f'{depth}' +
            '\nheads=' + f'{heads}' +
            '\nmlp_dim' + f'{mlp_dim}' +
            '\nchannels' + f'{channels}' +
            '\ndim_head=' + f'{dim_head}'
              )
        self.trans_classifier = Encoder(
            sequence_length=seq_len,
            patch_length=patch_size,
            num_classes=num_classes,
            token_dim=dim,
            num_blocks=depth,
            num_heads=heads,
            hidden_size=mlp_dim,
            channels=channels,
            dim_head=dim_head
        )

    def forward(self,input):
        input = input.permute(0,2,1)
        output  = self.trans_classifier(input)


        return output



class model_transformer_HAR(nn.Module):
    def __init__(self,batchsize,
                 seq_len=250,
            patch_size=10,
            num_classes=8,
            dim=512,
            depth=6,
            heads=8,
            mlp_dim=1024,
            channels=6,
            dim_head=64):
        super(model_transformer_HAR,self).__init__()
        print(
            'structure:transformer_HAR'+
            '\nsequence_length=' + f'{seq_len}, ' +
            '\npatch_size='+f'{patch_size}, '+
              '\nbatchsize='+f'{batchsize}, '+
              '\ndim='+f'{dim}'+
            '\ndepth=' + f'{depth}' +
            '\nheads=' + f'{heads}' +
            '\nmlp_dim' + f'{mlp_dim}' +
            '\nchannels' + f'{channels}' +
            '\ndim_head=' + f'{dim_head}'
              )
        self.trans_classifier = Encoder(
            sequence_length=seq_len,
            patch_length=patch_size,
            num_classes=num_classes,
            token_dim=dim,
            num_blocks=depth,
            num_heads=heads,
            hidden_size=mlp_dim,
            channels=channels,
            dim_head=dim_head
        )

    def forward(self,input):
        input = input.permute(0,2,1)
        output  = self.trans_classifier(input)
        return output




class model_HAR(nn.Module):
    def __init__(self,batchsize,device,hidden_size,num_layers,bidirectional,window_size):
        super(model_HAR,self).__init__()
        print('structure:LSTM_HAR, '+'\nbidirectional='+ f'{bidirectional}, '+'batchsize='+f'{batchsize}, '+'hidden_size='+f'{hidden_size}')
        self.rnn = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.batch_size = batchsize
        self.coeffienct = 1 if bidirectional == False else 2
        self.hidden_size = hidden_size
        self.window_size = window_size
        #input[Batchsize,windows length, 6]
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
        nn.BatchNorm1d(num_features=self.coeffienct*hidden_size),
        self.linears.append(nn.BatchNorm1d(num_features=self.coeffienct*hidden_size)),
        self.linears.append(nn.Linear(self.coeffienct*hidden_size,self.coeffienct*hidden_size))
        nn.BatchNorm1d(num_features=self.coeffienct*hidden_size),
        self.linears.append(nn.BatchNorm1d(num_features=self.coeffienct*hidden_size)),
        self.linears.append(nn.Linear(self.coeffienct*hidden_size,8))
        self.linears.append(nn.BatchNorm1d(num_features=8)),
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


class model_gru_HAR(nn.Module):
    def __init__(self,batchsize,device,hidden_size,num_layers,bidirectional,window_size):
        super(model_gru_HAR,self).__init__()
        print('structure:GRU_HAR'+'\nbidirectional='+ f'{bidirectional}, '+'batchsize='+f'{batchsize}, '+'hidden_size='+f'{hidden_size}')
        self.rnn = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.batch_size = batchsize
        self.coeffienct = 1 if bidirectional == False else 2
        self.hidden_size = hidden_size
        self.window_size = window_size
        #input[Batchsize,windows length, 6]
        self.rnn.append(nn.GRU(input_size=6,hidden_size=hidden_size,num_layers=num_layers,bidirectional=bidirectional,dropout=0.2))
        ##输出[B,250,1或2xhidden_size]
        self.h0 = torch.randn(self.coeffienct*num_layers,self.batch_size,hidden_size).to(device)##(num_layers*num_direction,batch,hidden_size)

        # self.rnn.append(nn.LSTM(input_size=2*32, hidden_size=16, num_layers=2, bidirectional=True, batch_first=True,dropout=0.2))
        # ##[B,250,2xhidden_size]
        # self.h1 = torch.randn(2 * 2, self.batch_size, 16)
        # self.c1 = torch.randn(2 * 2, self.batch_size, 16)
        self.linears.append(nn.Flatten())
        self.linears.append(nn.Linear(self.coeffienct*hidden_size*self.window_size,self.coeffienct*hidden_size))
        self.linears.append(nn.BatchNorm1d(num_features=self.coeffienct*hidden_size)),
        self.linears.append(nn.ReLU())
        self.linears.append(nn.Linear(self.coeffienct*hidden_size,self.coeffienct*hidden_size))
        self.linears.append(nn.BatchNorm1d(num_features=self.coeffienct*hidden_size)),
        self.linears.append(nn.Tanh())
        self.linears.append(nn.Linear(self.coeffienct*hidden_size,8))
        self.linears.append(nn.BatchNorm1d(num_features=8)),
        self.linears.append(nn.Softmax(dim=1))
        ##output [B,12]
    def forward(self,input):
        input = input.permute(1,0,2)
        output,_ = self.rnn[0](input,self.h0)
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







