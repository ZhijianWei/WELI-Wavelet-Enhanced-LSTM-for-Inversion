论文撰写中,训练数据暂不提供

## CWT类：
WaveletTransform 类用于执行小波变换，并保存第4和第5小波尺度对应的小波特征,成熟期前:4,成熟期:5
## process类：
DataPreprocessing 类用于加载数据、缩放数据、转换为监督学习格式以及分割数据。
## lstmModel 类:
LSTMModel用于构建、训练和预测LSTM模型。
## 辐射传输模型类：
CartesianProduct 类用于生成参数的笛卡尔积。
ProsailModel 类用于运行PROSAIL-D模型，并保存模拟结果到CSV文件。<br><br>
## main：
用户输入反射率Excel文件路径,并手动输入 Cab、LAI、tts 和 tto。<br><br>
1.调用 CWT 类进行小波变换。<br><br>
2.调用 preprocess 类进行特征工程;<br><br>
3.调用 LSTMModel 类进行模型训练和预测。<br><br>
4.调用 ProsailModel 类运行辐射传输模型，生成氮碳吸收加强的冠层光谱,并反演Nstr、Cm 和 lidfa。模拟结果保存到CSV文件中<br><br>


。
