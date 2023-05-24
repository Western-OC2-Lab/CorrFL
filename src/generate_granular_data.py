from this import d
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from joblib import Parallel, delayed, cpu_count

# This changes the time format of the original data
def change_time_format(index_value):
    date_format_str = '%Y-%m-%d %H:%M:%S'
    return datetime.strptime(index_value, date_format_str)

# This function finds the missing data points and interpolates them with the median value of each feature
def median_data(data):
    data.index = range(data.shape[0])
    median_values = data.median()
    for index_val in median_values.index:
        null_values_col = data.loc[:, index_val].isnull()
        data.loc[null_values_col, index_val] = median_values[index_val]
        
    return data

# This describes the history time window, the original dataset dir, and the destination dir after pre-processing
history = 5
dataset_dir = "../datasets/dirty_data/room00_preprocessed"
dst_dir = f"../datasets/dirty_data/room00_gran{history}_fine"

set_input_vars = {
    "node_920": ['humidity', 'temperature', 'pressure'],
    "node_924": ['humidity', 'temperature', 'pressure', 'co2'],
    "node_913": ['humidity', 'temperature', 'pressure', 'pir_cnt'],
    "node_914": ['humidity', 'temperature', 'pressure', 'pir_cnt'], 
    "node_915": ['humidity', 'temperature', 'pressure', 'pir_cnt'], 
    "node_916": ['humidity', 'temperature', 'pressure', 'pir_cnt']
}

# These thresholds are defined by the original dataset publishers
set_min_values = {
    'temperature': 0, 'humidity': 0, 'co2': 300, 'pressure': 950, 'pir_cnt': 0
}
set_max_values = {
    'temperature': 40, 'humidity': 99, 'co2': 5000, 'pressure': 1200, 'pir_cnt': 12
}


# This code is adopted from the original paper published in https://zenodo.org/record/3774723#.ZF06OXbMKUl and is resposible for the feature engineering step
def create_granular_dt(node_name, history):
    
    xframe = []
    inputvars = set_input_vars[node_name]

    df_node = pd.read_csv(f"{dataset_dir}/{node_name}.csv", index_col=[0])
    df_node.drop(columns = ['nodeid'], inplace=True)
    for col in df_node.columns:
        size_before, size_after = None, None
        if col == 'temperature':
            size_before = df_node.shape
        df_node = df_node[(df_node[col] >= set_min_values[col]) & (df_node[col] <= set_max_values[col])]
        if col == 'temperature':
            size_after = df_node.shape
            print(f'size before {size_before}, size after {size_after}')
    df_node = median_data(df_node)
    df_node.index = range(df_node.shape[0])
    idx = 1
    start_time = -1
    df = pd.DataFrame()
    while True:
        start_time=start_time + 1
        end_time = start_time + (history-1)
        if idx % 10000 == 0:
            print(f'{idx} out of {df_node.index[-1]}')
        if end_time + (history-1) > df_node.shape[0]-1 or end_time >= df_node.shape[0]:
            break
        
        sample_df = df_node.loc[start_time:end_time, :]
        sample_df.index = range(sample_df.shape[0])
        xframe, set_columns = [], []
        for i in range(sample_df.shape[0]):
            new_x = sample_df.loc[i, sample_df.columns].values
            set_cols = [c + '_{}'.format(i) for c in sample_df.columns]
            xframe.extend(new_x)
            set_columns.extend(set_cols)
        xframe = pd.DataFrame([xframe], columns=set_columns)

        diffs = xframe.loc[:, [c+ '_0' for c in sample_df.columns]].values - xframe.loc[:, [c+ '_1' for c in sample_df.columns]].values
        diffs2 = xframe.loc[:, [c+ '_0' for c in sample_df.columns]].values - xframe.loc[:, [c+ '_' + str(history-1) for c in sample_df.columns]].values
        
        diffs = pd.DataFrame(diffs, columns = [c + '_0_change' for c in sample_df.columns]) 
        diffs2 = pd.DataFrame(diffs2, columns = [c + '_'+str(history-1)+'_change' for c in sample_df.columns]) 
        
        xframe = pd.concat((xframe, diffs, diffs2), axis = 1)

        if node_name == 'node_924':
            yframe = df_node.loc[end_time + (history - 1), 'co2']
            y_df = pd.DataFrame([yframe], columns = ['co2'])
            xframe = pd.concat([xframe, y_df], axis = 1)
        df = pd.concat([df, xframe])
        idx += 1
    df.index = range(df.shape[0])
    df.to_csv(f"{dst_dir}/{node_name}.csv")

    return f"done with {node_name}"

# The feature engineering aspect is executed in parallel to facilitate the process. 
values = Parallel(n_jobs=int(cpu_count()/2), verbose=10)(delayed(create_granular_dt)(node_name, history) for node_name in ["node_920", "node_924", "node_913", "node_914", "node_915", "node_916"])



