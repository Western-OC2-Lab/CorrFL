import tqdm
from tqdm import tqdm
from server import *
import os
import numpy as np

dataset_dir, single_epoch, validation_epoch = "../datasets/dirty_data/room00_gran5_fine", 15, 3
hidden_dim_1, hidden_dim_2 = 128, 64
    
# Here, the Autoencoder parameters are defined. We restricted ourseleves to an Autoencoder with two hidden layers
ae_params = {
        'hidden_dim_1': hidden_dim_1,
        'hidden_dim_2': hidden_dim_2,
    }

set_small_epochs = [single_epoch] # This variable represents the number of communication cycles (every 15 CC is equivalent to 1 epoch for training data)
validation_epochs = [1] # The number of validation epochs
freq_models = [10] # Corresponds to Model Disptach Frequency (MDF)
all_combinations = np.array(np.meshgrid(set_small_epochs, validation_epochs)).T.reshape(-1, 2)

for combination in tqdm(all_combinations):

    for freq_model in freq_models:

        for iteration in range(3):
            SE, VE = combination[0], combination[1]
            run = SE * 1 # This variable corresponds to the instance when communication issues transpire. In our experimental procedure, we consider a cut-off after the completion of CorrFL training process. 
            name_exp = f"corrfl_arch_4-SE{SE}-R-{run}-VE-{VE}-FM-{freq_model}-IR-{iteration}" # The naming convention of the experiment
            print(f'Currently at {name_exp}')

            figure_dir = f"../results/figures/{name_exp}"
            stats_dir = f"../results/stats/Cross_dts_extreme/{name_exp}"
            models_dir = f"../results/models/{name_exp}"
            insights_dir = f"../results/insights/{name_exp}"

            os.makedirs(figure_dir, exist_ok=True)
            os.makedirs(stats_dir, exist_ok=True)
            os.makedirs(models_dir, exist_ok=True)
            os.makedirs(insights_dir, exist_ok=True)
            

            dir_params = {
                'dataset_dir': dataset_dir,
                'figure_dir': figure_dir,
                'models_dir': models_dir,
                'insights_dir':insights_dir
            }

            training_info = {
                    'data_size': 14*1440, # 14 days
                    'test_size': 0.25,
                    'after_epoch': run,
                    'figure_epoch': single_epoch,
                    'l1_loss': 0.1,
                    'l2_loss': 1.0 - 0.1,
                    'freq_models': freq_model,
                    'feature': 'pir_cnt', 
                    'feature_value': 8.0
            }

            server = Server(name_exp, ae_params, dir_params, training_info)

            for epoch in tqdm(range(1, SE+1)):
                
                server.train(epoch)

            # We only considered one absent sensor
            for null_sensor in ["node_924"]:

                print(f'Started the validation process {null_sensor}')

                for epoch in tqdm(range(1, VE+1)):
                    server.validate(epoch, null_sensor)

                print(f'Started the testing process {null_sensor}')
                server.test(0, null_sensor)
                stats_csv = f"{stats_dir}/{name_exp}_{null_sensor}.csv"
                pct_change = f"{stats_dir}/pct_{name_exp}_{null_sensor}.csv"
                validation_res = f"{stats_dir}/validation_{name_exp}_{null_sensor}.csv"
                testing_res = f"{stats_dir}/testing_{name_exp}_{null_sensor}.csv"
                corr_res = f"{stats_dir}/corr_{name_exp}_{null_sensor}.csv"

                server.result_df.to_csv(stats_csv)
                server.pct_change.to_csv(pct_change)
                server.validation_results.to_csv(validation_res)
                server.testing_results.to_csv(testing_res)
                server.set_corr_stats.to_csv(corr_res)



        

    
   

    

