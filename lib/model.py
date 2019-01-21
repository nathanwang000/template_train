import torch, math
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import warnings

MODELS = {}

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.features = nn.Sequential( # input 11x94x50
            nn.Conv2d(11, 32, kernel_size=3), # 92x48
            nn.MaxPool2d(kernel_size=2), # 46x24
            nn.ReLU(inplace=True),            
            nn.Conv2d(32, 32, kernel_size=3), # 44x22
            nn.MaxPool2d(kernel_size=2), # 22x11
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3), # 20x9
            nn.MaxPool2d(kernel_size=2), # 10x4
            nn.ReLU(inplace=True)
        )

        self.flatten = nn.Sequential(
            nn.Linear(32 * 10 * 4, 400),
            nn.ReLU(inplace=True),
            nn.Linear(400, 1)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.flatten(x)
        return x

class MLP(nn.Module):

    def __init__(self, neuron_sizes, activation=nn.LeakyReLU, bias=True): 
        super(MLP, self).__init__()
        self.neuron_sizes = neuron_sizes
        
        layers = []
        for s0, s1 in zip(neuron_sizes[:-1], neuron_sizes[1:]):
            layers.extend([
                nn.Linear(s0, s1, bias=bias),
                activation()
            ])
        
        self.classifier = nn.Sequential(*layers[:-1])

    def forward(self, x):
        x = x.view(-1, self.neuron_sizes[0])
        return self.classifier(x)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, use_gpu = False, output_dim = 1):
        super(LSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_size = int(input_size), hidden_size = int(hidden_size), num_layers = int(num_layers), dropout = float(dropout), batch_first = True)
        self.use_gpu = use_gpu 

        #Deal with Pytorch initialization for LSTMs being flawed
        for name,param in self.lstm.named_parameters():
            if 'weight' in name: 
                init.kaiming_uniform(param)
        
        #TODO: Layer Norm?
        
        self.fc1 = nn.Linear(hidden_size, 50)
        self.fc2 = nn.Linear(50, 1)
        self.relu = nn.ReLu(dim=2)
        self.float = self.FloatTensor
        if use_gpu:
            self.float = torch.cuda.FloatTensor
            
    def _get_final_y(self, output, batch_size, lengths):
        '''Given output from linear layer, find the corresponding last timestep output'''
        final_dist = []
        for i in range(batch_size):
            #Deals with uneven lengths in input
            final_dist.append(output[i, lengths[i]-1].view(1, 1, self.output_dim))
        return torch.cat(final_dist).view(batch_size, 1, self.output_dim)
    
    def forward(self, input, lengths):
        input = input.cpu().numpy()
        lengths = lengths.cpu().numpy() 
        #Find lengths of all intervals
        perm_index = reversed(sorted(range(len(lengths)), key=lambda k: lengths[k]))
        
        #Sort the input by length in order to pad, pack sequence for LSTM
        input = [torch.tensor(input[i]).view(-1, 22) for i in perm_index]
        lengths = list(reversed(sorted(lengths)))
        padded = rnn.pad_sequence(input, batch_first = True).view(len(input), max(lengths), 1)
        pack_padded = rnn.pack_padded_sequence(padded, lengths, batch_first = True)
        
        if self.use_gpu:
            pack_padded = pack_padded.cuda()

        #Get batch size, sequence length
        batch_size = pack_padded.batch_sizes[0]
        sequence_length = len(pack_padded.batch_sizes)
        
        #Initialize hidden state, cell state
        h0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size).type(self.float), requires_grad = False)
        c0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size).type(self.float), requires_grad = False)
        
                
        lstm_output, h_and_c = self.lstm(pack_padded, (h0, c0))

        #Unpack pad packed sequence, run through linear layer      
        unpacked, lens = rnn.pad_packed_sequence(lstm_output,batch_first = True)
        y = self.fc1(unpacked)
        
        #Deal with uneven lengths
        y_intermediate = self.relu(self._get_final_y(y, batch_size, lengths))
        
        y_final = self.fc2(y_intermediate)
        
        return y_final


MODELS['MLP'] = MLP
MODELS['CNN'] = CNN
MODELS['LSTM'] = LSTM

