import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
from matplotlib import pyplot
from math import sqrt

class LSTMModel:
    def __init__(self, n_hours=3, n_features=8):
        self.n_hours = n_hours
        self.n_features = n_features
        self.model = None

    def build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(50, input_shape=(self.n_hours, self.n_features)))
        self.model.add(Dense(4))
        self.model.compile(loss='mae', optimizer='adam')

    def train_model(self, train_X, train_y, test_X, test_y):
        history = self.model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.show()

    def predict(self, test_X, test_y, scaler):
        yhat = self.model.predict(test_X)
        inv_yhat = np.concatenate((test_X[:, :self.n_features], yhat), axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:, -4:]
        test_y = test_y.reshape((len(test_y), 4))
        inv_y = np.concatenate((test_X[:, :self.n_features], test_y), axis=1)
        inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:, -4:]
        rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
        print('Test RMSE: %.3f' % rmse)
        return inv_y, inv_yhat

    def save_results(self, inv_y, inv_yhat):
        results = pd.DataFrame({'Actual_Nstr': inv_y[:, 0], 'Actual_Cm': inv_y[:, 1], 'Actual_Cw': inv_y[:, 2], 'Actual_lidfa': inv_y[:, 3],
                                'Predicted_Nstr': inv_yhat[:, 0], 'Predicted_Cm': inv_yhat[:, 1], 'Predicted_Cw': inv_yhat[:, 2], 'Predicted_lidfa': inv_yhat[:, 3]})
        results.to_csv(r"predicted_results.csv", index=False) #替换成你自己的路径
        print("预测结果已成功保存到predicted_results.csv")
