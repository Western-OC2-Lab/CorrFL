import copy
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils import plot_predictions
from models import MLPLagged


'''
This class represents the local agents and includes all the operations taking place to train and evaluate their local models using 
the data collected on each node.
'''
class Client:
    
    def __init__(self, idx, input_size, size_data, df_client_train, df_client_validation, df_client_test, figure_dir, freq_model, figure_epoch):
        self.input_size = input_size
        self.size_data = size_data
        self.figure_dir = figure_dir
        self.freq_model = freq_model
        self.figure_epoch = figure_epoch

        nn = self.create_nn_model()
        self.idx=idx
        self.set_set(nn)

        self.optimizer = torch.optim.Adam(self.net.parameters(),lr=5e-3)
        
        self.loss_function = torch.nn.MSELoss()
        self.eval_loss = torch.nn.L1Loss()
        self.df_train = df_client_train
        self.df_validation = df_client_validation

        self.df_test = df_client_test
        
        self.log_interval = 200

    # This function initializes the neural network weights
    def init_normal(self, m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight.data, 0, 0.02)

    # This function updates the neural network weights
    def update_nn_parameters(self, new_params):
        self.net.load_state_dict(copy.deepcopy(new_params), strict=True)

    '''
    This function takes the epoch number, the name of the datasets (node), and a testing loader of the other data. This function enables 
    the local model to be evaluate using data from other nodes. 
    It returns the MSE and MAE loss following the evaluation process
    '''
    def test_other_dt(self, epoch, name_dt, other_test_loader):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.net.eval()
        # predicted_, targets_ = [], []
        predicted_, targets_ = torch.Tensor(), torch.Tensor()
        
        test_loader = other_test_loader
        with torch.no_grad():
            for (features, labels) in test_loader:
                outputs = self.net(features)
                predicted_ = torch.cat((predicted_, outputs.cpu()), dim = 0)
                targets_ = torch.cat((targets_, labels.cpu()), dim = 0)

        
        predicted_ = predicted_.reshape(-1,)
        targets_ = targets_.reshape(-1,)

        loss_mse = self.loss_function(targets_, predicted_).item()
        loss_mae = self.eval_loss(targets_, predicted_).item()
        loss_mae, loss_mse = round(loss_mae, 2), round(loss_mse, 2)
        
        if epoch % self.figure_epoch == 0:
            plot_predictions(predicted_, targets_, f"{self.figure_dir}/{self.idx}_{name_dt}_{epoch}.png", f"{self.idx}_{name_dt}_{epoch}_{loss_mae}")
                
        return loss_mae, loss_mse
    
    ''' 
    This function returns the training dataset based on the epoch number, which is equivalent to a Communication Cycle (CC), as defined in the paper. 
    In this code, the CC is dictated by the (self.size_data) variable.
    '''
    def retrieve_train_loader(self, epoch):
        start_index = (epoch) * (self.size_data) % (self.df_train.shape[0])
        end_index = (epoch + 1) * (self.size_data) % (self.df_train.shape[0])
        
        if end_index < start_index:
            start_index = 0
            end_index = self.size_data
        training_set = self.df_train.iloc[start_index:end_index]
        
        train_X, train_Y = training_set.drop(columns = ['co2']).values, training_set.co2.values
        
        X_torch_train, y_torch_train = torch.from_numpy(train_X).float(), torch.from_numpy(train_Y).float()
        training_dataset = TensorDataset(X_torch_train, y_torch_train)
        
        tensor_train_dataset = DataLoader(training_dataset, batch_size=8, shuffle=True)
        
        return tensor_train_dataset

    # Returns the validation loader
    def retrieve_validation_loader(self):
        
        train_X, train_Y = self.df_validation.drop(columns = ['co2']).values, self.df_validation.co2.values
        
        X_torch_train, y_torch_train = torch.from_numpy(train_X).float(), torch.from_numpy(train_Y).float()
        training_dataset = TensorDataset(X_torch_train, y_torch_train)
        
        tensor_train_dataset = DataLoader(training_dataset, batch_size=8, shuffle=True)
        
        return tensor_train_dataset
    
    # Returns the testing loader
    def retrieve_test_loader(self):
        train_X, train_Y = self.df_test.drop(columns = ['co2']).values, self.df_test.co2.values
        
        X_torch_train, y_torch_train = torch.from_numpy(train_X).float(), torch.from_numpy(train_Y).float()
        training_dataset = TensorDataset(X_torch_train, y_torch_train)
        
        tensor_train_dataset = DataLoader(training_dataset, batch_size=8, shuffle=True)
        
        return tensor_train_dataset
    
    '''
      Trains the local model based on the provided epoch (CC), which translates to retrieving the corresponding training data of this CC.
      At each Model Dispatch Frequency (self.freq_model), the model weights are appended to a local variable (set_params)
      Returns: 
      - the training time of the local model in this CC
      - the MAE loss
      - the accumulated model weights
    '''
    def train(self, epoch):
        self.net.train()
        
        train_data_loader = self.retrieve_train_loader(epoch)
        set_params = []
        running_loss, running_mae = 0.0, 0.0
        start_time = time.time()
        for i, (inputs, labels) in enumerate(train_data_loader):
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            outputs = torch.reshape(outputs.cpu(), (-1,))
            labels = torch.reshape(labels, (-1,))
            loss = self.loss_function(outputs, labels)
            mae_loss = self.eval_loss(outputs, labels)
            loss.backward()
        
            self.optimizer.step()
            
            running_loss+= loss.item()
            running_mae += mae_loss.item()
            if i % self.freq_model == 0 and i != 0:
                set_params.append(copy.deepcopy(self.get_nn_parameters()))
#                 print('[%d, %s, %2d] MSE loss %.3f, MAE loss %.3f' % (epoch, self.idx, i, running_loss / ((i+1)*len(outputs)), running_mae / ((i+1)*len(outputs))))
        end_time = time.time()
        total_train_time = round((end_time - start_time) / 60, 4)
        loss_mae = (running_mae) / (len(train_data_loader))
        return total_train_time, loss_mae, set_params

    '''
    In the validation phase, the models that are available continue their training process on the validation set. The code logic of this function
    resembles the train function, with similar returned variables.
    '''
    def validate(self, epoch):
        self.net.train()
        validation_data_loader = self.retrieve_validation_loader()
        set_params = []
        running_loss, running_mae = 0.0, 0.0
        start_time = time.time()
        for i, (inputs, labels) in enumerate(validation_data_loader):
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            outputs = torch.reshape(outputs.cpu(), (-1,))
            labels = torch.reshape(labels, (-1,))
            loss = self.loss_function(outputs, labels)
            mae_loss = self.eval_loss(outputs, labels)
            loss.backward()
        
            self.optimizer.step()
            
            running_loss+= loss.item()
            running_mae += mae_loss.item()
            if i % self.freq_model == 0 and i != 0:
                set_params.append(copy.deepcopy(self.get_nn_parameters()))
        end_time = time.time()
        total_train_time = round((end_time - start_time) / 60, 4)
        loss_mae = (running_mae) / (len(validation_data_loader))
        return total_train_time, loss_mae, set_params
    
    # This function evaluates the local model's performance on the testing set. It returns the MAE and MSE losses and the inference time.
    def test(self, epoch, name):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.net.eval()
        # predicted_, targets_ = [], []
        predicted_, targets_ = torch.Tensor(), torch.Tensor()
        
        test_loader = self.retrieve_test_loader()
        start_time = time.time()
        with torch.no_grad():
            for (features, labels) in test_loader:
                outputs = self.net(features)
                # targets_.extend(labels.cpu().numpy())
                # predicted_.extend(outputs.cpu().numpy())
                predicted_ = torch.cat((predicted_, outputs.cpu()), dim = 0)
                targets_ = torch.cat((targets_, labels.cpu()), dim = 0)

        end_time = time.time()
        inference_time = round((end_time - start_time) / 60, 2)
        predicted_ = predicted_.reshape(-1,)
        targets_ = targets_.reshape(-1,)

        # targets_ = torch.reshape(torch.Tensor(targets_), (-1,))
        # predicted_ = torch.reshape(torch.Tensor(predicted_), (-1,))


        loss_mse = self.loss_function(targets_, predicted_).item()
        loss_mae = self.eval_loss(targets_, predicted_).item()
        loss_mae, loss_mse = round(loss_mae, 2), round(loss_mse, 2)
        
        if epoch % self.figure_epoch == 0:
            plot_predictions(predicted_, targets_, f"{self.figure_dir}/{self.idx}_{name}_{epoch}.png", f"{self.idx}_{name}_{epoch}_{loss_mae}")
                
        return loss_mae, loss_mse, inference_time
    
    
    def get_nn_parameters(self):
        return self.net.state_dict()
    
    def create_nn_model(self):
        nn_model = MLPLagged(self.input_size)
        nn_model.apply(self.init_normal)
        return nn_model
    
    def set_set(self, net):
        self.net = net