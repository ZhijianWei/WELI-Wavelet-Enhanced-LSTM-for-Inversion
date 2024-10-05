import numpy as np
import pandas as pd
from pandas import DataFrame, concat
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessing:
    def __init__(self, n_hours=3, n_features=8):
        self.n_hours = n_hours  # 设置时间步长
        self.n_features = n_features  # 设置特征数量
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def load_data(self, filepath):
        dataset = pd.read_excel(filepath, header=0, index_col=0)
        return dataset.values  # 返回数据集的值

    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape  # 获取特征数量
        df = DataFrame(data)
        cols, names = list(), list()
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))  # 向列表cols中添加一个df.shift(i)的数据
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        for i in range(0, n_out):
            cols.append(df.shift(-i))  # 向列表cols中添加一个df.shift(-i)的数据
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        agg = concat(cols, axis=1)  # 将列表中多个DataFrame按列拼接起来
        agg.columns = names
        if dropnan:
            agg.dropna(inplace=True)  # 删除空值
        return agg

    def preprocess_data(self, values):
        scaled = self.scaler.fit_transform(values)  # 缩放特征值
        reframed = self.series_to_supervised(scaled, self.n_hours, 1)  # 转换为监督学习格式
        return reframed.values  # 返回转换后的值

    def split_data(self, values):
        n_train_hours = 365 * 24  # 设置训练集的大小
        train = values[:n_train_hours, :]
        test = values[n_train_hours:, :]
        n_obs = self.n_hours * self.n_features  # 计算观察值数量
        train_X, train_y = train[:, :n_obs], train[:, -4:]
        test_X, test_y = test[:, :n_obs], test[:, -4:]
        train_X = train_X.reshape((train_X.shape, self.n_hours, self.n_features))  # 重塑训练集输入
        test_X = test_X.reshape((test_X.shape, self.n_hours, self.n_features))  # 重塑测试集输入
        return train_X, train_y, test_X, test_y
