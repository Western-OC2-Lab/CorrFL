from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from model_trainer import ModelTrainer
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from client import *
import numpy as np
import os

'''
    This class defines the processes executed on the aggregation server. 
'''

class Server:
    
    '''
        ae_params = {
            'hidden_dim_1'
            'hidden_dim_2'
        },
        dir_params = {
            'dataset_dir',
            'figure_dir',
            'models_dir'
        },
        training_info = {
            'data_size': defining the smaller epoch (equivalent to CC)
            'test_size': size of testing data,
            'after_epoch': determines the delay of the server-side training (equivalent to d in the paper)
            'figure_epoch': determines the epoch at which figures are generated,
            'l1_loss': The weight for l1_loss, (For future work)
            'l2_loss': The weight for l2_loss
            'freq_models': interval of batches to send to the server (equivalent to MDF)
        }
    '''
    def __init__(self, name_exp, ae_params, dir_params, training_info):

        self.figure_dir, self.dataset_dir = dir_params['figure_dir'], dir_params['dataset_dir']

        self.hidden_dim_1, self.hidden_dim_2 = ae_params['hidden_dim_1'], ae_params['hidden_dim_2']
        self.name_exp = name_exp
        

        self.models_dir = dir_params['models_dir']
        self.data_size = training_info['data_size']
        self.test_size = training_info['test_size']
        self.after_epoch = training_info['after_epoch']
        self.figure_epoch = training_info['figure_epoch']
        self.feature_value = training_info['feature_value']
        self.feature = training_info['feature']
        self.insights_dir = dir_params['insights_dir']

        self.set_alphas = {}
        self.set_alphas['l1_loss'] = training_info['l1_loss']
        self.set_alphas['l2_loss'] = training_info['l2_loss'] 
        self.freq_models = training_info['freq_models']

        self.clients = self.create_clients(self.dataset_dir) # Here, we define the clients / local agents
        self.parameters = {} # This variable stores the set of weights of each set of nodes
        self.corr_fl = ModelTrainer([448, 336, 448], self.hidden_dim_1, self.hidden_dim_2,
        self.set_alphas, self.models_dir) #Here, we create the CorrFL wrapper class
        self.result_df = pd.DataFrame() # This DataFrame stores the performance results of all nodes and CorrFL methodology (MAE / training times)
        self.pct_1 = pd.DataFrame()
        self.pct_2 = pd.DataFrame()
        self.pct_3 = pd.DataFrame()
        self.pct_change = pd.DataFrame() # This dataframe stores the mean and std percentage change in the models produced by the CorrFL versus the ones that are given as inputs to CorrFL
        self.validation_results, self.testing_results = pd.DataFrame(), pd.DataFrame() # These DataFrames store the training and validation results (training time and MAE for validation, MAE for testing with and without CorrFL) 
        self.set_corr_stats = pd.DataFrame() # This dataframe stores the correlation statistics between the hidden representations
        
        # The variable that stores the nodes' weights is initialized as a map
        for client in self.clients.keys():
            self.parameters[self.clients[client].idx] = []
            
    def create_complementary_data(self, data_1, data_2):
        new_df = pd.DataFrame()
        for col in data_1.columns:
            if col in data_2.columns:
                new_df.loc[:, col] = data_2.loc[:, col].values
            else:
                new_df.loc[:, col] = data_1.loc[:, col].values
        return new_df

    def retrieve_test_loader(self, test_data):
        train_X, train_Y = test_data.drop(columns = ['co2']).values, test_data.co2.values
        
        X_torch_train, y_torch_train = torch.from_numpy(train_X).float(), torch.from_numpy(train_Y).float()
        training_dataset = TensorDataset(X_torch_train, y_torch_train)
        
        tensor_train_dataset = DataLoader(training_dataset, batch_size=8)
        
        return tensor_train_dataset

    def reset_weights_to_training(self):
        self.corr_fl.update_nn_parameters(self.set_weight_corr_fl)
        for client_id in self.clients.keys():
            self.clients[client_id].update_nn_parameters(self.set_weights[client_id])
        self.results_df, self.pct_change, self.validation_results, self.testing_results, self.set_corr_stats = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame() 

    '''
        This function describes the validation process, wherby each available node trains on the validation set. The CorrFL outputs the weights for 
        each model by calling its `test` function. 
        The absent models are evaluated based on their performance on the validation dataset and the results are retained under the names: `{null_sensor}_mae_validation_ABS`
        and `{null_sensor}_mae_validation_ABS_server`
    '''
    def validate(self, epoch, null_sensor):
        select_sensors, dict_result = ["node_920", "node_924"], {}

        if epoch == 1:
            self.sensor_weights = copy.deepcopy(self.clients[null_sensor].get_nn_parameters())
            self.set_weights = {client_id:copy.deepcopy(self.clients[client_id].get_nn_parameters()) for client_id in self.clients.keys() if client_id != null_sensor}
            self.set_weight_corr_fl = copy.deepcopy(self.corr_fl.get_nn_parameters())
            for client_id in self.clients.keys():

                self.parameters[client_id] = []

        
        len_params = 0
        for client_id in self.clients.keys():
            if client_id != null_sensor:
                training_time, client_mae, set_params = self.clients[client_id].validate(epoch)
                len_params  = len(set_params)
                dict_result[f'{client_id}_mae_validation'] = client_mae
                dict_result[f'{client_id}_validation_time'] = training_time
                self.parameters[client_id].extend(set_params)

        validation_data_loader = self.clients[null_sensor].retrieve_validation_loader()
        validation_mae, _ = self.clients[null_sensor].test_other_dt(epoch, "validation_no_server", validation_data_loader)
        dict_result[f'{null_sensor}_mae_validation_ABS'] = validation_mae
        set_app = []
        for i in range(len_params):
            set_app.append({'fc1.weight': np.zeros_like(self.sensor_weights['fc1.weight'].cpu()), 'fc2.weight': np.zeros_like(self.sensor_weights['fc2.weight'].cpu())})
        self.parameters[null_sensor].extend(set_app)
        (nn1_fc1_weights, nn1_fc2_weights), (nn2_fc1_weights, nn2_fc2_weights), (nn3_fc1_weights, nn3_fc2_weights) = self.corr_fl.test(self.parameters, null_sensor)

        for client in self.clients.keys():
            self.parameters[client] = []

        if null_sensor == "node_920":
            set_nn2_weights = {'fc1.weight': torch.Tensor(nn2_fc1_weights), 'fc2.weight': torch.Tensor(copy.deepcopy(self.clients[null_sensor].get_nn_parameters()['fc2.weight']))}
            def calculate_pct_change(orig_input_value, output_value):
                set_values = np.array(abs(np.divide((np.subtract(orig_input_value, output_value)), orig_input_value +1e-5)))
                return np.mean(set_values)*100, np.std(set_values)*100
            # change_clients = self.clients[null_sensor].copy()
            change_clients = Client(self.clients[null_sensor].idx, self.clients[null_sensor].input_size, self.clients[null_sensor].size_data, self.clients[null_sensor].df_train, self.clients[null_sensor].df_validation, self.clients[null_sensor].df_test, self.clients[null_sensor].figure_dir, self.clients[null_sensor].freq_model, self.clients[null_sensor].figure_epoch)
            change_clients.update_nn_parameters(set_nn2_weights)
            param_orig = np.array(self.clients[null_sensor].get_nn_parameters()['fc1.weight'].flatten().detach().numpy())
            param_new =  np.array(change_clients.get_nn_parameters()['fc1.weight'].flatten().detach().numpy())
            df_orig = pd.DataFrame([param_orig], columns = [f"nn_{i}" for i in range(param_orig.shape[0])])
            df_new = pd.DataFrame([param_new], columns = [f"nn_{i}" for i in range(param_new.shape[0])])

            if epoch == 1:
                df_orig.to_csv(f"{self.insights_dir}/fc1_orig_{null_sensor}_validation.csv")
            df_new.to_csv(f"{self.insights_dir}/fc1_new_{null_sensor}_{epoch}_validation.csv")
            validation_data_loader = change_clients.retrieve_validation_loader()
            validation_mae, _ = change_clients.test_other_dt(epoch, f"validation_{null_sensor}_server", validation_data_loader)
            dict_result[f'{null_sensor}_mae_validation_ABS_server'] = validation_mae
            
        elif null_sensor == "node_924":
            set_nn3_weights = {'fc1.weight': torch.Tensor(nn3_fc1_weights), 'fc2.weight':  torch.Tensor(copy.deepcopy(self.clients[null_sensor].get_nn_parameters()['fc2.weight']))}
            def calculate_pct_change(orig_input_value, output_value):
                set_values = np.array(abs(np.divide((np.subtract(orig_input_value, output_value)), orig_input_value +1e-5)))
                return np.mean(set_values)*100, np.std(set_values)*100
            # change_clients = self.clients[null_sensor].copy()
            change_clients = Client(self.clients[null_sensor].idx, self.clients[null_sensor].input_size, self.clients[null_sensor].size_data, self.clients[null_sensor].df_train, self.clients[null_sensor].df_validation, self.clients[null_sensor].df_test, self.clients[null_sensor].figure_dir, self.clients[null_sensor].freq_model, self.clients[null_sensor].figure_epoch)
            change_clients.update_nn_parameters(set_nn3_weights)
            param_orig = np.array(self.clients[null_sensor].get_nn_parameters()['fc1.weight'].flatten().detach().numpy())
            param_new =  np.array(change_clients.get_nn_parameters()['fc1.weight'].flatten().detach().numpy())
            df_orig = pd.DataFrame([param_orig], columns = [f"nn_{i}" for i in range(param_orig.shape[0])])
            df_new = pd.DataFrame([param_new], columns = [f"nn_{i}" for i in range(param_new.shape[0])])

            if epoch == 1:
                df_orig.to_csv(f"{self.insights_dir}/fc1_orig_{null_sensor}_validation.csv")
            df_new.to_csv(f"{self.insights_dir}/fc1_new_{null_sensor}_{epoch}_validation.csv")

            validation_data_loader = change_clients.retrieve_validation_loader()
            validation_mae, _ = change_clients.test_other_dt(epoch, f"validation_{null_sensor}_server", validation_data_loader)
            dict_result[f'{null_sensor}_mae_validation_ABS_server'] = validation_mae
        print(f"validation_res-server{dict_result[f'{null_sensor}_mae_validation_ABS_server']}")
        print(f"validation_res-Noserver{dict_result[f'{null_sensor}_mae_validation_ABS']}")

        self.validation_results = pd.concat([self.validation_results, pd.DataFrame.from_dict([dict_result])])
    
    # Similar to the validation function implementation, except that the present clients are trained on their validation dataset
    def test(self, epoch, null_sensor):
        set_weights = {client_id:copy.deepcopy(self.clients[client_id].get_nn_parameters()) for client_id in self.clients.keys()}
        set_weight_corr_fl = copy.deepcopy(self.corr_fl.get_nn_parameters())
        select_sensors, dict_result = ["node_920", "node_924"], {}
        if epoch == 0:
            for client_id in self.clients.keys():
                torch.save(self.clients[client_id].net.state_dict(), f"{self.models_dir}/{client_id}_validation_{null_sensor}.pt")

            torch.save(self.corr_fl.net.state_dict(), f"{self.models_dir}/corrNet_validation_{null_sensor}.pt")

        for client_id in self.clients.keys():
            curr_client = self.clients[client_id]
            if client_id != null_sensor:
                client_mae, _, _= self.clients[client_id].test(epoch, "testing")
                dict_result[f'{client_id}_mae_test'] = client_mae
                self.parameters[client_id].extend([copy.deepcopy(self.clients[client_id].get_nn_parameters())])
            else:
                client_mae, _, _ = self.clients[client_id].test(epoch, "testing")
                dict_result[f'{client_id}_mae_test_ABS'] = client_mae
                self.parameters[client_id].extend([{'fc1.weight': np.zeros_like(set_weights[client_id]['fc1.weight']), 'fc2.weight': np.zeros_like(set_weights[client_id]['fc2.weight'])}])
        (nn1_fc1_weights, nn1_fc2_weights), (nn2_fc1_weights, nn2_fc2_weights), (nn3_fc1_weights, nn3_fc2_weights) = self.corr_fl.test(self.parameters, null_sensor)

        if null_sensor == "node_920":
            set_nn2_weights = {'fc1.weight': torch.Tensor(nn2_fc1_weights), 'fc2.weight': torch.Tensor(copy.deepcopy(self.clients[null_sensor].get_nn_parameters()['fc2.weight']))}
            change_client = Client(self.clients[null_sensor].idx, self.clients[null_sensor].input_size, self.clients[null_sensor].size_data, self.clients[null_sensor].df_train, self.clients[null_sensor].df_validation, self.clients[null_sensor].df_test, self.clients[null_sensor].figure_dir, self.clients[null_sensor].freq_model, self.clients[null_sensor].figure_epoch)

            change_client.update_nn_parameters(set_nn2_weights)
            client_mae, _, _ = change_client.test(epoch, "testing_null")
            dict_result[f'{null_sensor}_mae_test_ABS_server'] = client_mae
        elif null_sensor == "node_924":
            set_nn3_weights = {'fc1.weight': torch.Tensor(nn3_fc1_weights), 'fc2.weight': torch.Tensor(copy.deepcopy(self.clients[null_sensor].get_nn_parameters()['fc2.weight']))}
            change_client = Client(self.clients[null_sensor].idx, self.clients[null_sensor].input_size, self.clients[null_sensor].size_data, self.clients[null_sensor].df_train, self.clients[null_sensor].df_validation, self.clients[null_sensor].df_test, self.clients[null_sensor].figure_dir, self.clients[null_sensor].freq_model, self.clients[null_sensor].figure_epoch)

            change_client.update_nn_parameters(set_nn3_weights)
            client_mae, _, _ = change_client.test(epoch, "testing_null")
            dict_result[f'{null_sensor}_mae_test_ABS_server'] = client_mae

        print(f"test_res-server{dict_result[f'{null_sensor}_mae_test_ABS_server']}")
        print(f"test_res-Noserver{dict_result[f'{null_sensor}_mae_test_ABS']}")
        
        self.testing_results = pd.concat([self.testing_results, pd.DataFrame.from_dict([dict_result])])

    def train_ae(self, epoch):
        dict_result = {}
        alt_models = [MLPLagged(28),MLPLagged(21),MLPLagged(28)]
        for client_id in self.clients.keys():
            training_time, client_mae, set_params = self.clients[client_id].train(epoch)
            dict_result[f'{client_id}_mae_train'] = client_mae
            dict_result[f'{client_id}_training_time'] = training_time
            self.parameters[client_id].extend(set_params)

            testing_result_mae, _ = self.clients[client_id].test(epoch, "self")
            dict_result[f'{client_id}_mae_test'] = testing_result_mae
            if epoch % self.figure_epoch == 0:
                print(f'eval epoch # {epoch} on #{client_id}: {client_mae, testing_result_mae}')

        if epoch >= self.after_epoch: 
            print('Currently Training the CorrNet')
            start_time = time.time()
            print(len(self.parameters["node_920"]))
            for client in self.clients.keys():
                if epoch == self.after_epoch:
                    df_fc1, df_fc2 = self.corr_fl.transform_to_df(self.parameters[client], "")
                    df_fc1.to_csv(f"../{self.name_exp}_fc1_{client}_training.csv")
                    df_fc2.to_csv(f"../{self.name_exp}_fc2_{client}_training.csv")

            server_loss, (nn1_fc1_weights, nn1_fc2_weights), (nn2_fc1_weights, nn2_fc2_weights), (nn3_fc1_weights, nn3_fc2_weights), stats_df = self.corr_fl.train(self.parameters, isValidationIdx = -1, ae_test=True)

            for client in self.clients.keys():
                if epoch == self.after_epoch:
                    df_fc1, df_fc2 = self.corr_fl.transform_to_df(self.parameters[client], "")
                    df_fc1.to_csv(f"../fc1_{client}_after_first_fedAvg.csv")
                    df_fc2.to_csv(f"../fc2_{client}_after_first_fedAvg.csv")

                self.parameters[client] = []
            end_time = time.time()

            corr_train_time = round((end_time - start_time) / 60, 3)
            dict_result['corr_time'] = corr_train_time
            dict_result['server_loss'] = server_loss
            print(f'server_loss {server_loss}, training_time {corr_train_time}')

            set_nn1_weights = {'fc1.weight': torch.Tensor(nn1_fc1_weights), 'fc2.weight': torch.Tensor(nn1_fc2_weights)}
            set_nn2_weights = {'fc1.weight': torch.Tensor(nn2_fc1_weights), 'fc2.weight': torch.Tensor(nn2_fc2_weights)}
            set_nn3_weights = {'fc1.weight': torch.Tensor(nn3_fc1_weights), 'fc2.weight': torch.Tensor(nn3_fc2_weights)}
            
            for type_client in ["node_913", "node_914", "node_915", "node_916"]:
                alt_models[0].load_state_dict(copy.deepcopy(set_nn1_weights), strict=True)

            alt_models[1].load_state_dict(copy.deepcopy(set_nn2_weights), strict=True)
            alt_models[2].load_state_dict(copy.deepcopy(set_nn3_weights), strict=True)

            for client_id in self.clients.keys():
                curr_client = self.clients[client_id]
                if client_id in ["node_913", "node_914", "node_915", "node_916"]:
                    curr_client.net = alt_models[0]
                elif client_id == "node_920":
                    curr_client.net = alt_models[1]
                else:
                    curr_client.net = alt_models[2]

                testing_result_mae, _ = curr_client.test(epoch, "self")

                dict_result[f'{client_id}_mae_server'] = testing_result_mae

                if epoch % self.figure_epoch == 0:
                    print(f'After server #{self.clients[client_id].idx} eval on #{epoch} is {testing_result_mae}')

           
            self.set_corr_stats = pd.concat([self.set_corr_stats, stats_df[0]])
            self.pct_change = pd.concat([self.pct_change, stats_df[1]])
        else:
            dict_result['server_loss'] = "N/A"
            dict_result['corr_time'] = "N/A"
            for client_id in self.clients.keys():
                dict_result[f'{client_id}_mae_server'] = "N/A"
                dict_result[f'{client_id}_mae'] = "N/A"
                
        self.result_df = pd.concat([self.result_df, pd.DataFrame.from_dict([dict_result])])

    '''
        This function defines the mundane Federated Learning paradigm without any service interruptions
    '''
    def validate_FL(self, epoch):
        dict_result = {}
        for client_id in self.clients.keys():
            training_time, client_mae, set_params = self.clients[client_id].validate(epoch)
            dict_result[f'{client_id}_mae_validation'] = client_mae
            dict_result[f'{client_id}_validation_time'] = training_time
            self.parameters[client_id].extend(set_params)


        (nn1_fc1_weights, nn1_fc2_weights), (nn2_fc1_weights, nn2_fc2_weights), (nn3_fc1_weights, nn3_fc2_weights) = self.corr_fl.FedAvg(self.parameters)
        set_nn1_weights = {'fc1.weight': torch.Tensor(nn1_fc1_weights), 'fc2.weight': torch.Tensor(nn1_fc2_weights)}
        set_nn2_weights = {'fc1.weight': torch.Tensor(nn2_fc1_weights), 'fc2.weight': torch.Tensor(nn2_fc2_weights)}
        set_nn3_weights = {'fc1.weight': torch.Tensor(nn3_fc1_weights), 'fc2.weight': torch.Tensor(nn3_fc2_weights)}
        
        for type_client in ["node_913", "node_914", "node_915", "node_916"]:
            self.clients[type_client].update_nn_parameters(set_nn1_weights)
            self.parameters[type_client].extend([set_nn1_weights])

        self.clients["node_920"].update_nn_parameters(set_nn2_weights)
        self.clients["node_924"].update_nn_parameters(set_nn3_weights)

        self.parameters["node_920"].extend([set_nn2_weights])
        self.parameters["node_924"].extend([set_nn3_weights])

        for client_id in self.clients.keys():
            client_mae, _, inference_time= self.clients[client_id].test(epoch, "testing")
            dict_result[f'{client_id}_mae_test_FedAvg'] = client_mae
            dict_result[f'{client_id}_mae_inference_time_FedAvg'] = inference_time
        
        self.validation_results = pd.concat([self.validation_results, pd.DataFrame.from_dict([dict_result])])

    '''
        This function defines the processes, whereby the local agents just continue their training without any aggregations from the server. 
    '''
    def train_FL(self, epoch, null_sensor):
        dict_result = {}
        dict_result['epoch'] = epoch
        for client_id in self.clients.keys():
            if client_id == "null_sensor":
                training_time, client_mae, set_params = self.clients[client_id].train(epoch)
                dict_result[f'{client_id}_mae_train'] = client_mae
                dict_result[f'{client_id}_training_time'] = training_time
                self.parameters[client_id].extend(set_params)

                testing_result_mae, _, _ = self.clients[client_id].test(epoch, "self")
                dict_result[f'{client_id}_mae_test'] = testing_result_mae
            
                if epoch % self.figure_epoch == 0:
                    print(f'After epoch #{epoch} eval on #{client_id}: {client_mae, testing_result_mae}')
        self.result_df = pd.concat([self.result_df, pd.DataFrame.from_dict([dict_result])])

    '''
        This function excecutes the different process that transpire during the training phase. These processes are as follows:
        - When all of the models are available (epoch < after_epoch), the local agents are trained. During this phase, the training time, the mae for training and testing
        sets, and the models' weights are all retained.
        - When one of the models are unavailable (epoch >= after_epoch), the CorrFL models starts its training process with all the accumulated model weights. This can be done 
        simultaneously while the local agents are training. The training function returns the corrFL's loss, and the set of weights for each set of nodes after applying Federated
        Averaging. 
        - The weights corresponding to the local models are removed and the testing results after the federated averaging process are retained. 

        NOTE: In the manuscript, we calculated the training times of the local agents to be equivalent to the average of the local times of all agents, considering that these
        agents run in parallel. While in real-world implementations, this assumption is valid, it is not implemented in this code. However, this feature can be easily integrated using 
        joblib's `parallel` package.  
    '''
    def train(self, epoch):
        dict_result = {}
        dict_result['epoch'] = epoch
        for client_id in self.clients.keys():
            training_time, client_mae, set_params = self.clients[client_id].train(epoch)
            dict_result[f'{client_id}_mae_train'] = client_mae
            dict_result[f'{client_id}_training_time'] = training_time
            self.parameters[client_id].extend(set_params)

            testing_result_mae, _, _ = self.clients[client_id].test(epoch, "self")
            dict_result[f'{client_id}_mae_test'] = testing_result_mae
            
            if epoch % self.figure_epoch == 0:
                print(f'After epoch #{epoch} eval on #{client_id}: {client_mae, testing_result_mae}')

        if epoch >= self.after_epoch: 
            print('Currently Training the CorrNet')
            print(len(self.parameters["node_920"]))
            for client in self.clients.keys():
                if epoch == self.after_epoch:
                    df_fc1, df_fc2 = self.corr_fl.transform_to_df(self.parameters[client], "")
                    df_fc1.to_csv(f"{self.insights_dir}/fc1_{client}_training.csv")
                    df_fc2.to_csv(f"{self.insights_dir}/fc2_{client}_training.csv")
            start_time = time.time()
            server_loss, (nn1_fc1_weights, nn1_fc2_weights), (nn2_fc1_weights, nn2_fc2_weights), (nn3_fc1_weights, nn3_fc2_weights), stats_df = self.corr_fl.train(self.parameters)

            for client in self.clients.keys():

                self.parameters[client] = []
            end_time = time.time()

            corr_train_time = round((end_time - start_time) / 60, 3)
            dict_result['corr_time'] = corr_train_time
            dict_result['server_loss'] = server_loss
            print(f'server_loss {server_loss}, training_time {corr_train_time}')

            set_nn1_weights = {'fc1.weight': torch.Tensor(nn1_fc1_weights), 'fc2.weight': torch.Tensor(nn1_fc2_weights)}
            set_nn2_weights = {'fc1.weight': torch.Tensor(nn2_fc1_weights), 'fc2.weight': torch.Tensor(nn2_fc2_weights)}
            set_nn3_weights = {'fc1.weight': torch.Tensor(nn3_fc1_weights), 'fc2.weight': torch.Tensor(nn3_fc2_weights)}
            
            for type_client in ["node_913", "node_914", "node_915", "node_916"]:
                self.clients[type_client].update_nn_parameters(set_nn1_weights)
                self.parameters[type_client].extend([set_nn1_weights])

            self.clients["node_920"].update_nn_parameters(set_nn2_weights)
            self.clients["node_924"].update_nn_parameters(set_nn3_weights)

            self.parameters["node_920"].extend([set_nn2_weights])
            self.parameters["node_924"].extend([set_nn3_weights])
            
            for client_id in self.clients.keys():
                if epoch == self.after_epoch:
                    df_fc1, df_fc2 = self.corr_fl.transform_to_df(self.parameters[client_id], "")
                    df_fc1.to_csv(f"{self.insights_dir}/fc1_{client_id}_after_first_fedAvg.csv")
                    df_fc2.to_csv(f"{self.insights_dir}/fc2_{client_id}_after_first_fedAvg.csv")
                testing_result_mae, _, _ = self.clients[client_id].test(epoch, "self")

                dict_result[f'{client_id}_mae_server'] = testing_result_mae

                if epoch % self.figure_epoch:
                    print(f'After server #{client_id} eval on #{epoch} is {testing_result_mae}')
           
            self.set_corr_stats = pd.concat([self.set_corr_stats, stats_df[0]])
            self.pct_change = pd.concat([self.pct_change, stats_df[1]])
        else:
            dict_result['server_loss'] = "N/A"
            dict_result['corr_time'] = "N/A"
            for client_id in self.clients.keys():
                dict_result[f'{client_id}_mae_server'] = "N/A"
            

                
        self.result_df = pd.concat([self.result_df, pd.DataFrame.from_dict([dict_result])])

    '''
    This function splits the datasets of all nodes into two types based on the feature that determines the splitting criteria. In our experimental 
    procedure, we determine any the activity level of more than 7.0 in any of the lagged features. After that, we return the union of all of these
    values. The indices with that fullfill this criteria are denoted as type 0 datasets while the ones who do not are denoted as type 1 datasets.
    '''        
    def split_datasets(self, dataset_dir, feature = 'pir_cnt', value = 8.0):
        all_loaded_df = {node_id_csv.split(".")[0]: pd.read_csv(f"{dataset_dir}/{node_id_csv}", index_col=[0]) for node_id_csv in os.listdir(dataset_dir) if node_id_csv != "aggregated_data.csv"}
        set_indices = []
        new_f = f'{feature}_0'
        for df_key in all_loaded_df.keys():
            if new_f in all_loaded_df[df_key].columns:
                new_dt = all_loaded_df[df_key]
                set_pir_columns = [col for col in new_dt.columns if col.startswith(feature) and col.endswith("_change") == False]
                returned_indices = [np.where(new_dt[col] >= value)[0] for col in set_pir_columns] # Have to change this into quantiles
                union_indices = returned_indices[0]
                for i in range(1, len(returned_indices)):
                    un = np.union1d(union_indices, returned_indices[i])
                    union_indices = un
                set_indices.append(union_indices)

        all_indices = set_indices[0]
        for i in range(1, len(set_indices)):
            un = np.union1d(all_indices, set_indices[i])
            all_indices = un
            
        for df_key in all_loaded_df.keys():
            all_loaded_df[df_key].loc[:, 'type'] = 1
            all_loaded_df[df_key].loc[all_indices, 'type'] = 0

        return all_loaded_df

    '''
    This function create the local agents by assoiciating it with a client class instance. It first creates the client for the node that captures the response variable so that it is available for other clients (nodes). For each client,
    their respective training and testing sets are defined based on the feature splitting criteria defined in split_dataset function. The training and testing sets are standardized. 
    Clients equivalent to the number of nodes are created, each with its unique set of training and testing datsets.
    The testing and validation datasets are split equally.  
    '''
    def create_clients(self, dataset_dir):
        print("Started creating the clients")
        clients = {}
        all_loaded_dfs = self.split_datasets(dataset_dir, 'pir_cnt', self.feature_value)

        main_df = all_loaded_dfs["node_924"]
        training_data, testing_data = main_df.loc[main_df.type == 1].drop(columns = ['type']), main_df.loc[main_df.type == 0].drop(columns = ['type'])
        print(np.mean(training_data.co2.values), np.std(training_data.co2.values))
        print(np.mean(testing_data.co2.values), np.std(testing_data.co2.values))

        scaler=StandardScaler()
        training_data_X = scaler.fit_transform(training_data.drop(columns = ['co2']))
        testing_data_X = scaler.transform(testing_data.drop(columns = ['co2']))
                
        training_data = pd.DataFrame(data = np.c_[training_data_X, training_data.co2.values], columns = training_data.columns)
        testing_data = pd.DataFrame(data = np.c_[testing_data_X, testing_data.co2.values], columns = testing_data.columns)
        test_data, valid_data = train_test_split(testing_data, test_size = 0.5, random_state=42)

        new_client = Client("node_924", len(training_data.columns)-1, self.data_size, 
        training_data, test_data, valid_data, self.figure_dir, self.freq_models, self.figure_epoch)
        clients['node_924'] = new_client
        output_vars= main_df.loc[:, 'co2'].values

        for df_key in all_loaded_dfs.keys():
            if df_key != "node_924":
                data = all_loaded_dfs[df_key]
                data = data.iloc[:output_vars.shape[0]]
                data.loc[:, 'co2'] = output_vars[:data.shape[0]]
                training_data, testing_data = data.loc[data.type == 1].drop(columns = ['type']), data.loc[data.type == 0].drop(columns = ['type'])

                scaler = StandardScaler()

                training_data_X = scaler.fit_transform(training_data.drop(columns = ['co2']))
                testing_data_X = scaler.transform(testing_data.drop(columns = ['co2']))
                
                training_data = pd.DataFrame(data = np.c_[training_data_X, training_data.co2.values], columns = training_data.columns)
                testing_data = pd.DataFrame(data = np.c_[testing_data_X, testing_data.co2.values], columns = testing_data.columns)
                test_data, valid_data = train_test_split(testing_data, test_size = 0.5, random_state=42)
                
                new_client = Client(df_key, len(training_data.columns)-1, 
                self.data_size, training_data, valid_data, testing_data, self.figure_dir, self.freq_models, self.figure_epoch)
                clients[df_key] = new_client

        print("Successfully created the clients")
        
        return clients