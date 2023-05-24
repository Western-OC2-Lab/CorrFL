import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

'''
This function plots models' predictions. The function takes the following parameters:
- predictions: np.array. The predictions of CO2 concentration values.
- y_values: np.array. The original set of CO2 concentration values.
- file_name: str. This parameter describes the directory and the name of the file whereby the figure is saved.
- title: str. The figure's title

'''
def plot_predictions(predictions, y_values, file_name, title):
    sns.set_theme()
    plt.figure(figsize=(30, 12))
    plt.plot(predictions, 'v', label='predictions', alpha=0.6)
    plt.plot(y_values, 'x', label='values', alpha=0.6)
    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)
    plt.xlabel('Timestamp', fontsize=  27)
    plt.ylabel('CO2 Levels', fontsize=  27)
    plt.title(title, fontsize = 32)
    plt.legend()
    plt.savefig(file_name)
    plt.close()

# This function returns the storage medium.
def initialize_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


'''
This function returns the training and testing sets of a dataframe after applying a scaling method. It takes the following parameters:
- df: pd.DataFrame: the dataframe used for training and testing purposes
- type_scaler: the scaling method.
- test_size: the size of the testing set.
'''
def retrieve_train_test_tensors(df, type_scaler, test_size=0.25):
    train_X, test_X, train_Y, test_Y = train_test_split(df.drop(columns=['co2']).values, df.loc[:, 'co2'].values, 
                                                        test_size=test_size, random_state=42)
    scaler = type_scaler
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)

    X_torch_train, X_torch_test = torch.from_numpy(train_X).float(), torch.from_numpy(test_X).float()
    y_torch_train, y_torch_test = torch.from_numpy(train_Y).float(), torch.from_numpy(test_Y).float()

    training_dataset = TensorDataset(X_torch_train, y_torch_train)
    testing_dataset = TensorDataset(X_torch_test, y_torch_test)

    tensor_train_dataset = DataLoader(training_dataset, batch_size=8)
    tensor_test_dataset = DataLoader(testing_dataset, batch_size=8)

    return tensor_train_dataset, tensor_test_dataset