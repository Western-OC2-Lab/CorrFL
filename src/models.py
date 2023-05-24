import torch
import torch.nn as nn

'''
MLPLagged defines the neural network model deployed on each local agent. The architecture is split into an input layer, hiiden layer, 
and output layer as per the paper's definition. 
Note: Both nn.ReLU() and functional.relu can be used. However, we opted to use nn.ReLU as it represents a module that can be added to the sequential layer.
'''
class MLPLagged(nn.Module):
    
    def __init__(self, input_size):
        super(MLPLagged, self).__init__()

        self.fc1 = nn.Linear(input_size, 16, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1, bias=False)


    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        return x
    
'''
Corr_FL is the neural network model proposed in the paper. The Corr_FL architecture is similar to a multi-view autoencoder that combines the 
latent space. 
The module is written in a way to enable different size inputs with a uniform 2-layered AE architecture.
'''

class Corr_FL(nn.Module):
    
    def __init__(self, input_sizes, hidden_dim_1, hidden_dim_2):
        super(Corr_FL, self).__init__()
        self.encoders = nn.ModuleList()
    
        for input_size in input_sizes:
            self.encoders.append(
                nn.Sequential(
                    nn.Linear(input_size, hidden_dim_1),
                    nn.Linear(hidden_dim_1, hidden_dim_2),
                ))
           
            
        self.decoders = nn.ModuleList()
        
        for input_size in input_sizes:
            self.decoders.append(nn.Sequential(
                    nn.Linear(hidden_dim_2, hidden_dim_1),
                    nn.Linear(hidden_dim_1, input_size),
                ))
            
            
    
    def forward(self, inputs):
        out = []
        for idx, enc in enumerate(self.encoders):
            out.append(enc(inputs[idx]))
            
        self.out = out
        
        common_rep = out[0]
        for i in range(1, len(out)):
            common_rep = torch.add(common_rep, out[idx])
        
        self.common_rep = common_rep
        
        reconstructed_layer = []
        
        for i in range(len(out)):
            reconstructed_layer.append(self.decoders[i](common_rep))
        
        return reconstructed_layer