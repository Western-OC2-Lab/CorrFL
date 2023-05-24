from symbol import single_input
from joblib import Parallel, delayed
from psutil import cpu_count
from torch.utils.data import DataLoader, TensorDataset
import copy
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from models import Corr_FL
import torch.nn as nn
import pandas as pd

'''
    This class is the wrapper class that executes all the main functions of the CorrFL model
'''
class ModelTrainer:
    
    '''
        input_sizes: The number of features for each model
        hidden_dim_1: The number of neurons of the first hidden layer of the Autoencoder
        hidden_dim_2: The number of neurons of the second hiddent layer of the Autoencoder
        set_weights: The weights corresponding for the L1 and L2 losses. (Placeholders)
        model_dir: The directory where the models are saved
    '''
    def __init__(self, input_sizes, hidden_dim_1, hidden_dim_2, set_weights, model_dir):
        self.input_sizes = input_sizes
        self.h_dim_1 = hidden_dim_1
        self.h_dim_2 = hidden_dim_2
        self.l1_loss_w = set_weights['l1_loss']
        self.l2_loss_w = set_weights['l2_loss']
        self.model_dir = model_dir

        self.orig_values = {"m1": None, "m2": None, "m3": None} # Saves the number of values for each model type
        self.set_scalers = {"m1": {"mean": None, "std": None} , "m2": {"mean": None, "std": None}, "m3": {"mean": None, "std": None}} # Saves the standardization values for each model 
        
        model_nn = self.ae_model()
        self.net = model_nn
        self.optimizer = torch.optim.Adam(self.net.parameters(),lr=1e-3)
        
        self.log_interval = 200

    '''
        This function returns the weights of the first and second hidden layers (fc1 and fc2) as dataframes. Toward that end, the 
        weights of the first layer are standardized and returned, if the flag with_scaling is True. 
        The standardization process is gradually applied to match the steady flow of data.
        The following inputs are required:
        - client_params: the set of weights
        - model_type: the type of model these weights belong to. In our case, it is either m1, m2, or m3.
        - with_scaling: a flag to determine if standardization process should be applied.
    '''
    def transform_to_df(self, client_params, model_type, with_scaling=False):
        df_fc1, df_fc2 = pd.DataFrame(), pd.DataFrame()

        def single_value_transformation(fc1_values, fc2_values):
            set_fc1, set_fc2 = np.array(fc1_values.flatten()), np.array(fc2_values.flatten())

            return set_fc1, set_fc2
        
        set_values = Parallel(n_jobs=int(cpu_count() / 2), verbose=0, max_nbytes=1e6)(delayed(single_value_transformation)(client_params[i]['fc1.weight'], client_params[i]['fc2.weight']) for i in range(len(client_params)))

        val_fc1 = [item[0] for item in set_values]
        val_fc2 = [item[1] for item in set_values]
        df_fc1 = pd.DataFrame(val_fc1, columns=[f"nn_{i}" for i in range(val_fc1[0].shape[0])])
        if with_scaling:
            if self.set_scalers[model_type]["mean"] is None:
                scaler = StandardScaler()
                df_fc1_scaled = scaler.fit_transform(df_fc1.values)
                self.orig_values[model_type] = df_fc1.shape[0]

                df_fc1 = pd.DataFrame(df_fc1_scaled, columns=[f"nn_{i}" for i in range(val_fc1[0].shape[0])])
                self.set_scalers[model_type]["mean"] = scaler.mean_
                self.set_scalers[model_type]["std"] = np.sqrt(scaler.var_+ 1e-5)

            else:
                scaler = StandardScaler()
                df_fc1_inter = scaler.fit_transform(df_fc1.values)
                len_new, len_old = df_fc1.shape[0], self.orig_values[model_type]
                target_mean, target_std = self.set_scalers[model_type]["mean"], self.set_scalers[model_type]["std"]
                print(f'old {model_type}-{target_mean[0]}-{target_std[0]}')
                self.set_scalers[model_type]["mean"] = ((len_new) * scaler.mean_ + (len_old) * (target_mean)) / ( len_new + len_old)
                self.set_scalers[model_type]["std"] = ((len_new) * np.sqrt(scaler.var_ + 1e-5) + (len_old) * (target_std)) / ( len_new + len_old)
                self.orig_values[model_type] += len_new

                new_target_mean , new_target_std = self.set_scalers[model_type]["mean"], self.set_scalers[model_type]["std"]
                print(f'new {model_type}-{new_target_mean[0]}-{new_target_std[0]}')

                new_cols = pd.DataFrame()
                for idx, col in enumerate(df_fc1.columns):
                    a_col = df_fc1.loc[:, col].apply(lambda x: (x-new_target_mean[idx]) / (new_target_std[idx]))

                    new_cols.loc[:, col] = a_col.copy()
                df_fc1 = new_cols.copy()
        df_fc2 = pd.DataFrame(val_fc2, columns=[f"nn_{i}" for i in range(val_fc2[0].shape[0])])

        df_fc1.index = range(df_fc1.shape[0])
        df_fc2.index = range(df_fc2.shape[0])
        
        return df_fc1, df_fc2

    def FedAvg(self, parameters):
        first_set_clients = {"node_913": {"fc1": None, "fc2": None}, "node_914":{"fc1": None, "fc2": None}, "node_915":{"fc1": None, "fc2": None}, "node_916":{ "fc1": None, "fc2": None}}
        second_set_clients = {"node_920": {"fc1": None, "fc2": None}}
        third_set_clients = {"node_924": {"fc1": None, "fc2": None}}


        for key in first_set_clients.keys():
            client_params = parameters[key]
            df_fc1, df_fc2 = self.transform_to_df(client_params, "m1")
            first_set_clients[key]["fc1"] = df_fc1
            first_set_clients[key]["fc2"] = df_fc2

        second_key = list(second_set_clients.keys())[0]
        df_fc1, df_fc2 = self.transform_to_df(parameters[second_key], "m2")
        second_set_clients[second_key]["fc1"] = df_fc1
        second_set_clients[second_key]["fc2"] = df_fc2

        third_key = list(third_set_clients.keys())[0]
        df_fc1, df_fc2 = self.transform_to_df(parameters[third_key], "m3")
        third_set_clients[third_key]["fc1"] = df_fc1
        third_set_clients[third_key]["fc2"] = df_fc2

        fc1_values, fc2_values = pd.DataFrame(), pd.DataFrame() 
        for key in first_set_clients.keys():
            fc1_values = pd.concat([fc1_values, first_set_clients[key]["fc1"].iloc[-1]], axis = 1)
            fc2_values = pd.concat([fc2_values, first_set_clients[key]["fc2"].iloc[-1]], axis = 1)

        return (fc1_values.mean(axis = 1).values.reshape(16, 28), fc2_values.mean(axis = 1).values.reshape(-1, 1).T), (second_set_clients["node_920"]["fc1"].iloc[-1].values.reshape(16, 21), second_set_clients["node_920"]["fc2"].iloc[-1].values.reshape(-1, 1).T), (third_set_clients["node_924"]["fc1"].iloc[-1].values.reshape(16, 28), third_set_clients["node_924"]["fc2"].iloc[-1].values.reshape(-1, 1).T)
    
    '''
        This function returns the training sets that will be used as inputs for the CorrFL model. It returns three sets:
        - The first that includes all the weights of the first hidden layers for all models (trainable parameters)
        - The second that includes all the weights of the second hidden layer for all models (non-trainable parameters)
        - The third that includes the last weights for all models
        This function includes the with_scaling flag so that it gives the flexibility to include the new weights in the calculation of the standardization parameters (validation_data will be with a False flag)

    '''
    def generate_train_dts(self, parameters, with_scaling, batch_size=16):
        first_set_clients = {"node_913": {"fc1": None, "fc2": None}, "node_914":{"fc1": None, "fc2": None}, "node_915":{"fc1": None, "fc2": None}, "node_916":{ "fc1": None, "fc2": None}}
        second_set_clients = {"node_920": {"fc1": None, "fc2": None}}
        third_set_clients = {"node_924": {"fc1": None, "fc2": None}}


        for key in first_set_clients.keys():
            client_params = parameters[key]
            df_fc1, df_fc2 = self.transform_to_df(client_params, "m1", with_scaling)
            first_set_clients[key]["fc1"] = df_fc1
            first_set_clients[key]["fc2"] = df_fc2

        second_key = list(second_set_clients.keys())[0]
        df_fc1, df_fc2 = self.transform_to_df(parameters[second_key], "m2", with_scaling)
        second_set_clients[second_key]["fc1"] = df_fc1
        second_set_clients[second_key]["fc2"] = df_fc2

        third_key = list(third_set_clients.keys())[0]
        df_fc1, df_fc2 = self.transform_to_df(parameters[third_key], "m3", with_scaling)
        third_set_clients[third_key]["fc1"] = df_fc1
        third_set_clients[third_key]["fc2"] = df_fc2

        fc1_values, fc2_values = pd.DataFrame(), pd.DataFrame() 
        list_keys = list(first_set_clients.keys())
        fc1_values = first_set_clients[list_keys[0]]["fc1"]
        fc2_values = first_set_clients[list_keys[0]]["fc2"]

        # for key in first_set_clients.keys():
        for i in range(1, len(list_keys)):
            fc1_values = fc1_values.add(first_set_clients[list_keys[i]]["fc1"])
            fc2_values = fc2_values.add(first_set_clients[list_keys[i]]["fc2"])

        
        fc1_values = fc1_values.div(len(list_keys))
        fc2_values = fc2_values.div(len(list_keys))

        all_parameters_trainable = [torch.Tensor(fc1_values.values), 
        torch.Tensor(second_set_clients["node_920"]["fc1"].values), torch.Tensor(third_set_clients["node_924"]["fc1"].values)]
        all_parameters_nontrainable = [torch.Tensor(fc2_values.iloc[-1].values.reshape(-1, 1).T), torch.Tensor(second_set_clients["node_920"]["fc2"].iloc[-1].values.reshape(-1, 1).T), torch.Tensor(third_set_clients["node_924"]["fc2"].iloc[-1].values.reshape(-1, 1).T)]

        data_loader = DataLoader([[all_parameters_trainable[0][i], all_parameters_trainable[1][i], all_parameters_trainable[2][i]] for i in range(len(all_parameters_trainable[0]))], batch_size = batch_size, shuffle=True)
        # set_data_loaders = []
        # for trainable_data in all_parameters_trainable:
        #     trainable_data = torch.Tensor(trainable_data)
        #     X_client_first = TensorDataset(trainable_data)
        #     dt_loader = DataLoader(X_client_first, batch_size)
        #     set_data_loaders.append(dt_loader)
        
        final_parameters = [torch.Tensor(fc1_values.iloc[-1].values), torch.Tensor(second_set_clients["node_920"]["fc1"].iloc[-1].values),torch.Tensor(third_set_clients["node_924"]["fc1"].iloc[-1].values)]

        return data_loader, all_parameters_nontrainable, final_parameters

    def average_nn_parameters(self, parameters):
        new_params = {}
        for name in parameters[0].keys():
            new_params[name] = sum([param[name].data for param in parameters]) / len(parameters)

        return new_params
    
    
    def ae_model(self):
        model_nn = Corr_FL(self.input_sizes, self.h_dim_1, self.h_dim_2)
        return model_nn
    
    # Calculates the reconstruction loss
    def calculate_l1_loss(self, ae_inputs, set_outputs, lossfn, isValidationIdx):
        l1_loss = 0.0
        for i in range(len(ae_inputs)):
            if i == isValidationIdx:
                continue
            # l1_loss += lossfn(ae_inputs[i], set_outputs[i]) / (ae_inputs[0].shape[1])
            l1_loss += lossfn(ae_inputs[i], set_outputs[i])
            # l1_loss += self.calculate_pct_change(ae_inputs[i], set_outputs[i])

        return l1_loss / (len(ae_inputs))

    # Calculates the reconstruction loss when one model is absent
    def calculate_l2_loss(self, net, set_inputs, loss_fn, isValidationIdx):
        r_rec_loss, set_hidden_rep, nb_values = 0.0, {}, 0
        if isValidationIdx != -1:
            with torch.no_grad():
                temp_output = net(set_inputs)
                hidden_rep = net.common_rep
                set_hidden_rep[f'input_{0}'] = hidden_rep.cpu()
                for idx in range(len(set_inputs)):
                    if idx == isValidationIdx:
                        continue
                    rel_output = temp_output[idx]
                    r_rec_loss += loss_fn(set_inputs[idx], rel_output)
                    nb_values += 1
        else:
            for i in range(len(set_inputs)):
                x_input = []
                for idx, input_batch in enumerate(set_inputs):
                    if i == idx:
                        x_input.append(torch.zeros_like(input_batch))
                    else:
                        x_input.append(input_batch)
                with torch.no_grad():
                    temp_output = net(x_input)
                    hidden_rep = net.common_rep
                    set_hidden_rep[f'input_{i}'] = hidden_rep.cpu()
                    for j in range(len(set_inputs)):
                        rel_output = temp_output[j]
                        r_rec_loss += loss_fn(set_inputs[j], rel_output)
                        # r_rec_loss += self.calculate_pct_change(set_inputs[j], rel_output)
                        nb_values += 1
        return r_rec_loss / (nb_values), set_hidden_rep

    def calculate_euclidean(self, h1, h2, lambda_val):
        return torch.mean(torch.cdist(h1, h2))

    # Calculates the correlation loss. 
    def calculate_l3_loss(self, an_inputs, type_loss, lambda_val =1):
        sum_combinations = 0.0
        dict_result = {}
        input_names = list(an_inputs.keys())
        if len(input_names) > 1:
            all_combinations = np.array(np.meshgrid(input_names, input_names)).T.reshape(-1, 2)

            for combination in all_combinations:
                input_1, input_2 = combination[0], combination[1]
                combination_name = "_".join([input_1, input_2])
                if input_1 == input_2:
                    continue
                corr_val = self.calculate_correlation(an_inputs[input_1], an_inputs[input_2], lambda_val)
                # corr_val = corr_val.item()
                dict_result[combination_name] = corr_val.item()
                re_val = None
                if type_loss == "h":
                    re_val = (1-corr_val) / abs(corr_val)
                    if corr_val < 0:
                        re_val *= 4
                    re_val = (-1/6) * (re_val)
                    
                else:
                    re_val = (1-corr_val) * (1/len(all_combinations))
                    
                sum_combinations +=  re_val

        return sum_combinations, pd.DataFrame.from_dict([dict_result])

    
    def produce_hidden_rep(self, set_inputs, loss_fn):
        r_l1_loss, set_hidden_rep = 0.0, {}
        temp_output = self.net(set_inputs)
        
        for i in range(len(set_inputs)):             
            rel_output = temp_output[i]
            r_l1_loss += loss_fn(set_inputs[i], rel_output)
            hidden_rep = self.net.out[i]
            set_hidden_rep[f'input_{i}'] = hidden_rep.cpu()
            
        return set_hidden_rep

    # Calculates the correlations between the hidden representations when one model is absent 
    def calculate_correlation(self, h1, h2, lambda_val):
        h1_mean, h2_mean = torch.mean(h1, axis=0), torch.mean(h2, axis=0)

        h1_centered, h2_centered = torch.subtract(h1, h1_mean), torch.subtract(h2, h2_mean)
        corr_nr = torch.sum(torch.multiply(h1_centered, h2_centered), axis=0)



        corr_dr1 = torch.sqrt(torch.sum(torch.square(h1_centered), axis=0) + 1e-8)

        corr_dr2 = torch.sqrt(torch.sum(torch.square(h2_centered), axis=0)+ 1e-8)


        corr_dr = torch.add(torch.multiply(corr_dr1, corr_dr2), 1e-5)

        corr = torch.divide(corr_nr, corr_dr)


        corr_mean = torch.add(torch.mean(corr), 1e-5)
        
        return corr_mean
    
    def reconstruction_loss(self, set_inputs, set_outputs, lossfn):
        l1_loss = 0.0
        for i in range(len(set_inputs)):
            l1_loss += lossfn(set_inputs[i], set_outputs[i])
            
        return l1_loss
                
    def update_nn_parameters(self, new_params):
        self.net.load_state_dict(copy.deepcopy(new_params), strict=True)

    def get_nn_parameters(self):
        return self.net.state_dict()

    # Applies standardization for the present models while the absent models are set to zeros
    def standardize_parameters(self, final_parameters, null_sensor):
        scaled_parameters = []
        for idx, prm in enumerate(final_parameters):
            scaled_parameter = torch.divide(torch.subtract(prm, torch.Tensor(self.set_scalers[f"m{idx+1}"]["mean"])), torch.Tensor(self.set_scalers[f"m{idx+1}"]["std"]))
            scaled_parameters.append(scaled_parameter)

        self.save_parameters()
        if null_sensor in ['node_914', 'node_915', 'node_916']:
            scaled_parameters[0] = torch.zeros_like(scaled_parameters[0])
        elif null_sensor == "node_920":
            scaled_parameters[1] = torch.zeros_like(scaled_parameters[1])
        elif null_sensor == "node_924":
            scaled_parameters[2] = torch.zeros_like(scaled_parameters[2])
        else:
            scaled_parameter = scaled_parameters

        return scaled_parameters

    def save_parameters(self):
        list_models = self.set_scalers.keys()
        for model in list_models:
            mean_model, std_model = self.set_scalers[model]["mean"], self.set_scalers[model]["std"]
            mean_df = pd.DataFrame([mean_model], columns = [f"nn_{i}" for i in range(mean_model.shape[0])])
            std_df = pd.DataFrame([std_model], columns = [f"nn_{i}" for i in range(std_model.shape[0])])
            mean_df.to_csv(f"{self.model_dir}/mean_{model}.csv")
            std_df.to_csv(f"{self.model_dir}/std_{model}.csv")

    # reverses the standardization process
    def reverse_standardization(self, final_parameters):
        scaled_parameters = []
        for idx, prm in enumerate(final_parameters):
            scaled_parameter = torch.add(torch.multiply(prm, torch.Tensor(self.set_scalers[f"m{idx+1}"]["std"])), torch.Tensor(self.set_scalers[f"m{idx+1}"]["mean"]))

            scaled_parameters.append(scaled_parameter)

        return scaled_parameters

    '''
        This function executes the CorrFL model in testing model. 
        The function first retrieves the most recent weights such that `with_scaling=False` so that the standardization parameters are not altered.
        Then, the weights are normalized based on the available standardization parameters. (null_sensor -> absent model -> zeors)
        The weights obtained after standardization are fed to the CorrFL models.
        A reverse standardization process is applied to obtain the new weights. 
    '''
    def test(self, parameters, null_sensor=""):
        data_loaders, non_trainable_params, final_parameters = self.generate_train_dts(parameters, with_scaling=False, batch_size=16)

        scaled_parameters = self.standardize_parameters(final_parameters, null_sensor)
        nn_1_weights, nn_2_weights, nn_3_weights, nw_outputs = None, None, None, None
        with torch.no_grad():
            nw_outputs = self.net(scaled_parameters)
            nn_1_weights, nn_2_weights, nn_3_weights = nw_outputs[0], nw_outputs[1], nw_outputs[2]

        normalized_outputs = self.reverse_standardization(nw_outputs)
        nn_1_weights = torch.Tensor(normalized_outputs[0].reshape(16, 28))
        nn_2_weights = torch.Tensor(normalized_outputs[1].reshape(16, 21))
        nn_3_weights = torch.Tensor(normalized_outputs[2].reshape(16, 28))

        return (nn_1_weights, non_trainable_params[0]), (nn_2_weights, non_trainable_params[1]), (nn_3_weights, non_trainable_params[2])
    
    def calculate_pct_change(self, orig_input_value, output_value):
        # set_values = np.array(abs(np.divide((np.subtract(orig_input_value.cpu(), output_value.cpu().detach().numpy())), orig_input_value.cpu().detach().numpy() +1e-5)))
        set_values = torch.abs(torch.divide((torch.subtract(orig_input_value.cpu(), output_value.cpu())), torch.add(orig_input_value.cpu(), 1e-5)))

        return torch.multiply(torch.mean(set_values), 100)


    '''
        This function trains the CorrFL model. It first retrieves the training data using the `generate_train_dts` function such that the scaling attribute is True. 
        Next, the CorrFL model is progressively trained using loss functions L1, L2, and L3. After the completion of the training process, this function returns the 
        weights after applying the Federated Averaging procedure (`self.FedAvg`).
        Information about the progression of the loss (`df_loss`) with training and the correlation between the hidden representations when models are absent
        (`set_corr_stats`) are retained.  
    '''
    def train(self, parameters, isValidationIdx = -1, ae_test=False):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        set_epochs = 15
        self.net = self.net.to(device)
        def calculate_pct_change(orig_input_value, output_value):
            set_values = np.array(abs(np.divide((np.subtract(orig_input_value.cpu().detach().numpy(), output_value.cpu().detach().numpy())), orig_input_value.cpu().detach().numpy() +1e-5)))
            return np.mean(set_values)*100, np.std(set_values)*100
            
        self.net.train()
        
        nn_1, nn_2, nn_3 = [], [], []
        pct1_s, pct2_s, pct3_s = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        dict_pct = {}
        pct_df = pd.DataFrame()

        set_corr_stats = pd.DataFrame()
        server_loss = 0.0
        data_loaders, non_trainable_params, final_parameters = self.generate_train_dts(parameters, with_scaling=True, batch_size=16)

        for epoch in range(0, set_epochs):
            running_loss = 0.0
            for i , all_data in enumerate(data_loaders):

                input_dt_1, input_dt_2, input_dt_3 = all_data[0], all_data[1], all_data[2]
                set_inputs = [input_dt_1.to(device), input_dt_2.to(device), input_dt_3.to(device)]
                
                self.optimizer.zero_grad()
                outputs = self.net(set_inputs)

                #l1_loss that minimizes the reconstruction error when other inputs are inexistent
                loss=nn.MSELoss()
                l1_loss = self.calculate_l1_loss(set_inputs, outputs, loss, isValidationIdx)
                
                l2_loss, set_hidden_rep = self.calculate_l2_loss(self.net, set_inputs, loss, isValidationIdx)
                l3_loss, df_loss = self.calculate_l3_loss(set_hidden_rep, "e")

                df_loss.loc[:, 'epoch'] = epoch
                set_corr_stats= pd.concat([set_corr_stats, df_loss])
                loss_values = l1_loss.cpu() + l2_loss.cpu() + l3_loss

                loss_values.backward()

                # for name, param in self.net.named_parameters():
                #     print('name, grad_param', name, param.grad)
                
                self.optimizer.step()
                
                running_loss += loss_values
            
            server_loss += running_loss / (i+1)
            print('[%d/%d] Loss: %.3f' %(epoch + 1, set_epochs, server_loss / (epoch + 1)))
            if epoch % 3 == 0:
                torch.save(copy.deepcopy(self.net.state_dict()),  f"{self.model_dir}/{epoch}_arch-5.pt")

        nn_1_weights, nn_2_weights, nn_3_weights, nw_outputs = None, None, None, None
        with torch.no_grad():
            self.net = self.net.to('cpu')
            # normalized_parameters = self.standardize_parameters(final_parameters)
            nw_outputs = self.net(final_parameters)
            nr_outputs = self.reverse_standardization(nw_outputs)

            for idx_out, out in enumerate(nr_outputs):
                mean_values, std_values = calculate_pct_change(final_parameters[idx_out], out)
                dict_pct[f'{idx_out}_mean'] = mean_values
                dict_pct[f'{idx_out}_std'] = std_values
            pct_df = pd.concat([pct_df, pd.DataFrame.from_dict([dict_pct])])

        normalized_outputs = self.reverse_standardization(nw_outputs)
        # output_df = pd.DataFrame([normalized_outputs[2]], columns = [f"nn_{i}" for i in range(normalized_outputs[2].shape[0])])
        # nn_output_df = pd.DataFrame([nw_outputs[2]], columns = [f"nn_{i}" for i in range(normalized_outputs[2].shape[0])])
        # output_df.to_csv("../weights_output.csv")
        # nn_output_df.to_csv("../nn_output.csv")

        nn_1_weights = torch.Tensor(normalized_outputs[0].reshape(16, 28))
        nn_2_weights = torch.Tensor(normalized_outputs[1].reshape(16, 21))
        nn_3_weights = torch.Tensor(normalized_outputs[2].reshape(16, 28))

        stats_df = [set_corr_stats, pct_df]
        server_loss = (server_loss / set_epochs)

        set_val_params = self.FedAvg(parameters)

        # if ae_test:
        #     return server_loss.item(), (nn_1_weights, set_val_params[0][1]), (nn_2_weights, set_val_params[1][1]), (nn_3_weights, set_val_params[2][1]), stats_df
        # else:
        if ae_test:
            return server_loss.item(), (nn_1_weights, set_val_params[0][1]), (nn_2_weights, set_val_params[1][1]), (nn_3_weights, set_val_params[2][1]), stats_df
        else:
            return server_loss.item(), (set_val_params[0][0], set_val_params[0][1]), (set_val_params[1][0], set_val_params[1][1]), (set_val_params[2][0], set_val_params[2][1]), stats_df


        # if isValidationIdx == -1:
        #     return server_loss.item(), set_val_params, stats_df
        # else:
        #     return server_loss.item(), (set_val_params[0][0], set_val_params[0][1]), (set_val_params[1][0], set_val_params[1][1]), (nn_3_weights, set_val_params[2][1]), stats_df

