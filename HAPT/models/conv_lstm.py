import torch
import torch.nn as nn
from inputpipeline import datasets
from models.transformer import transformer_encoder

class Conv_lstm(nn.Module):
    def __init__(self,device='cpu',num_lstm_layers=2,hidden_size=48,batch_size=32):
        super(Conv_lstm, self).__init__()
        print('structure:Conv_lstm'+'\nbidirectional='+ f'{True}, '+'batchsize='+f'{batch_size}, '+'hidden_size='+f'{hidden_size}')
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=6,out_channels=12,kernel_size=5,stride=1,padding='same',),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=12,out_channels=24,kernel_size=5,stride=1,padding='same',),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
## [ 32 , 24 , 62 ]
            # nn.Conv1d(in_channels=24, out_channels=48, kernel_size=5, stride=1, padding='SAME', ),
            # nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2),

        )
        self.rnn = nn.ModuleList()
        self.rnn.append(nn.LSTM(input_size=24,
                    hidden_size=48,
                    num_layers=2,
                    bidirectional=True,
                    dropout=0.2))
        self.h0 = torch.randn( 2 * 2, batch_size , 48).to(device)
        ##(num_layers*num_direction,batch,hidden_size)
        self.c0 = torch.randn( 2 * num_lstm_layers, batch_size, hidden_size).to(device)
        self.lstm_act = nn.Tanh()

        self.linears = nn.Sequential(
            nn.Flatten(),
            nn.Linear( 2 * hidden_size * 62, 2 * hidden_size),
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
# input = torch.randn(32,6,250)
# output = mdl(input)