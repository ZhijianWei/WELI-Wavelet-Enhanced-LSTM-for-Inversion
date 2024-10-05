from data_preprocessing import DataPreprocessing
from lstm_model import LSTMModel
import pandas as pd
import numpy as np
from keras.models import load_model

if __name__ == "__main__":
    # 加载和预处理数据
    data_preprocessing = DataPreprocessing()
    filepath = ".xlsx"
    values = data_preprocessing.load_data(filepath)
    reframed_values = data_preprocessing.preprocess_data(values)

    train_X, train_y, test_X, test_y = data_preprocessing.split_data(reframed_values)

    model = load_model("lstm_model.h5")

    # 恢复训练
    lstm_model = LSTMModel()
    lstm_model.model = model
    lstm_model.train_model(train_X, train_y, test_X, test_y)

    # 保存模型
    lstm_model.model.save("lstm_model_resumed.h5")
