import pandas as pd
import numpy as np
import torch
import sys
from PLEModel import StackedPLE, PLEVanilla, setup_seed
from torch.utils.data import Dataset, DataLoader
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def percentage_error(actual, predicted):
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred)))) * 100


class Mydataset(Dataset):

    # Initialization
    def __init__(self, data, label, mode='2D'):
        self.data, self.label, self.mode = data, label, mode

    # Get item
    def __getitem__(self, index):
        if self.mode == '2D':
            return self.data[index, :], self.label[index, :]
        elif self.mode == '3D':
            return self.data[:, index, :], self.label[:, index, :]

    # Get length
    def __len__(self):
        if self.mode == '2D':
            return self.data.shape[0]
        elif self.mode == '3D':
            return self.data.shape[1]

class PLETestingModel(BaseEstimator, RegressorMixin):

    def __init__(self, input_dim, specific_experts, shared_experts, experts_out, experts_hidden, towers_hidden, tasks_num,
                 output_dim=1, n_epoch=200, batch_size=64, lr=0.001, device=torch.device('cuda:0'), seed=1024):
        super(PLETestingModel, self).__init__()

        # Set seed
        torch.manual_seed(seed)

        # Parameter assignment
        self.input_dim = input_dim
        self.specific_experts = specific_experts
        self.shared_experts = shared_experts
        self.experts_out = experts_out
        self.experts_hidden = experts_hidden

        self.towers_hidden = towers_hidden
        self.tasks_num = tasks_num

        self.output_dim = output_dim

        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.seed = seed

        # Initialize Scaler
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()


        # Model Instantiation
        self.loss_hist = []
        self.model = PLEVanilla(input_dim=input_dim, specific_experts=specific_experts,
                                shared_experts=shared_experts, experts_out=experts_out,
                                experts_hidden=experts_hidden, towers_hidden=towers_hidden, tasks_num=tasks_num).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.criterion = nn.MSELoss(reduction='mean')

    def fit(self, X, y):
        X_scaled = self.scaler_X.fit_transform(X[:, :2])
        y = self.scaler_y.fit_transform(y)

        y = y.reshape(-1, self.output_dim)


        X_3d = np.hstack([X_scaled, X[:, -1].reshape((-1, 1))])
        y_3d = y

        print('Let us print the shapes {}, {}'.format(X_3d.shape, y_3d.shape))


        dataset = Mydataset(torch.tensor(X_3d, dtype=torch.float32, device=self.device),
                            torch.tensor(y_3d, dtype=torch.float32, device=self.device), '2D')

        self.model.train()
        for i in range(self.n_epoch):
            self.loss_hist.append(0)
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle = True)
            for batch_X, batch_y in data_loader:
                batch_X, batch_index = batch_X[:, :2],  batch_X[:, -1].reshape((-1, 1)).type(torch.int64)

                batch_y = batch_y



                self.optimizer.zero_grad()

                output, _ = self.model(batch_X)

                output = torch.cat(output, dim=-1)
                # print(output.shape)
                # print(batch_index.shape)
                # print(batch_index)
                # sys.exit(0)
                # print(batch_index.shape, batch_index.dtypes)

                final_output = output.gather(1, batch_index)



                loss = self.criterion(final_output, batch_y)

                self.loss_hist[-1] += loss.item()

                loss.backward()

                self.optimizer.step()

            print('Epoch:{}, Loss:{}'.format(i + 1, self.loss_hist[-1]))
        print('Optimization finished')

        return self


    def predict(self, X):

        X_input = self.scaler_X.transform(X[:, :2])
        X_indicate = X[:, -1].reshape((-1, 1))




        X = torch.tensor(X_input, dtype=torch.float32, device=self.device)
        X_indicate = torch.tensor(X_indicate, dtype=torch.int64, device=self.device)

        self.model.eval()
        with torch.no_grad():

            y, _ = self.model(X)
            y_final = torch.cat(y, dim=-1)
            y_final = y_final.gather(1, X_indicate)

            # 放上cpu转为numpy
            y_final = y_final.cpu().numpy()
        y = self.scaler_y.inverse_transform(y_final)

        return y

if __name__ == "__main__":
    data = pd.read_csv('testSampleForRegression.csv', sep=',', header=0, index_col=0)
    print(data)
    SEED = 1024
    setup_seed(seed=SEED)
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = PLEVanilla(input_dim=2, specific_experts=3,
                       shared_experts=2, experts_out=4,
                       experts_hidden=6, towers_hidden=6, tasks_num=3).to(DEVICE)

    X_data = data.iloc[:, :3].values.reshape((-1, 3))
    y_data = data.iloc[:, -1].values.reshape((-1, 1))

    mdl = PLETestingModel(input_dim=2, specific_experts=3, shared_experts=2,
                          experts_out=4,  experts_hidden=6, towers_hidden=6, tasks_num=3, output_dim=1,
                          n_epoch=50, batch_size=64, lr=0.001, device=torch.device('cuda:0'), seed=SEED).fit(X_data, y_data)




    test_data = pd.read_csv('testData.csv', index_col=0, header=0, sep=',')
    test_x = test_data.iloc[:, :3].values.reshape((-1, 3))
    test_y = test_data.iloc[:, -1].values.reshape((-1, 1))

    y_pred_test = mdl.predict(X=test_x)
    print(y_pred_test.shape)
    print(test_y.shape)

    test_rmse = np.sqrt(mean_squared_error(y_pred_test, test_y))
    test_r2 = r2_score(y_pred_test, test_y)
    test_mae = mean_absolute_error(y_pred_test, test_y)
    test_mape = mean_absolute_percentage_error(y_true=y_pred_test, y_pred=test_y)
    print('test_rmse = ' + str(round(test_rmse, 5)))
    print('test_r2 = ', str(round(test_r2, 5)))

    print('seed = ', SEED, ', test_rmse = ', round(test_rmse, 5), ', r2 = ', round(test_r2, 5), ', mape = ',
          round(test_mape, 5), ', mae = ', round(test_mae, 5))



