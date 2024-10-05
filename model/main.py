from CWT_model import WaveletTransform
from preprocess import DataPreprocessing
from lstm_model import LSTMModel
from prosail_model import ProsailModel
import pandas as pd
import numpy as np

if __name__ == "__main__":
    # 用户输入反射率Excel文件路径
    filepath = input("请输入反射率Excel文件路径：")

    # 小波变换
    CWT_model = WaveletTransform()
    df_WF4, df_WF5 = CWT_model.transform(pd.read_excel(filepath))

    # 数据特征工程
    preprocess = DataPreprocessing()
    values = np.hstack([df_WF4['WF4'].values.reshape(-1, 1), df_WF5['WF5'].values.reshape(-1, 1)])
    reframed_values = preprocess.preprocess_data(values)
    
    train_X, train_y, test_X, test_y = preprocess.split_data(reframed_values)

    # 构建和训练LSTM模型，预测结果
    lstm_model = LSTMModel()
    lstm_model.build_model()
    lstm_model.train_model(train_X, train_y, test_X, test_y)

    inv_y, inv_yhat = lstm_model.predict(test_X, test_y, preprocess.scaler)
    lstm_model.save_results(inv_y, inv_yhat)

    # 提取预测结果中的Nstr, Cm, lidfa
    predicted_results = pd.read_csv("predicted_results.csv")
    N = predicted_results['Predicted_Nstr'].values
    Cm = predicted_results['Predicted_Cm'].values
    lidfa = predicted_results['Predicted_lidfa'].values

    # 手动输入Cab, LAI, tts和tto
    Cab = float(input("请输入叶片叶绿素含量 (Cab): "))
    LAI = float(input("请输入叶面积指数 (LAI): "))
    tts = float(input("请输入太阳天顶角 (tts): "))
    tto = float(input("请输入观测天顶角 (tto): "))
    
    # 其他固定参数
    Car = Cab / 4
    Cb = [0.]
    Cw = [0.]
    hspot = [0.1]
    psi = [0.] #观测天顶角，默认垂直观测
    psoil1 = [0.]

    # 运行辐射传输模型
    prosail_model = ProsailModel(N, Cab, Car, Cb, Cw, Cm, LAI, lidfa, hspot, tts, tto, psi, psoil1)
    prosail_model.Cm_index = 5  # 设置Cm参数索引
    results = prosail_model.run_simulations()
    prosail_model.save_results(results)
