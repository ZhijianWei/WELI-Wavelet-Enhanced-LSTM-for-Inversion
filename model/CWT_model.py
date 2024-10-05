import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['axes.unicode_minus'] = False  # 确保负号显示正确

class WaveletTransform:
    def __init__(self, wavelet='cmor1.5-1.0', scales=np.arange(1, 128)):
        self.wavelet = wavelet  # 设置小波函数
        self.scales = scales    # 设置小波尺度范围

    def transform(self, data):
        wavelengths = data.iloc[:, 0].values
        reflectance = data.iloc[:, 1].values
        [cfs, frequencies] = pywt.cwt(reflectance, self.scales, self.wavelet)

        # 提取第4和第5小波尺度对应的小波特征
        WF4 = cfs[3, :]
        WF5 = cfs[4, :]

        df_WF4 = pd.DataFrame({'Wavelength': wavelengths, 'WF4': WF4})
        df_WF5 = pd.DataFrame({'Wavelength': wavelengths, 'WF5': WF5})

        df_WF4.to_csv(r"WF4.csv", index=False) #替换成你直接的路径
        df_WF5.to_csv(r"WF5.csv", index=False)

        print("小波特征已成功保存到WF4.csv和WF5.csv")
        return df_WF4, df_WF5  # 返回WF4和WF5的DF



    # 小波特征的三维坐标图展示
    def visualize(self, df_WF4, df_WF5):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制第4尺度的三维散点图
        X = df_WF4['Wavelength'].values
        Y = np.array( * len(X))  # 第4尺度
        Z = df_WF4['WF4'].values


        sc = ax.scatter(X, Y, Z, c=Z, cmap='viridis', depthshade=True, label='Scale 4')

        # 绘制第5尺度的三维散点图
        X = df_WF5['Wavelength'].values
        Y = np.array( * len(X))  # 第5尺度
        Z = df_WF5['WF5'].values


        sc = ax.scatter(X, Y, Z, c=Z, cmap='plasma', depthshade=True, label='Scale 5')

        # 设置坐标轴范围和标签
        ax.set_xlabel('Wavelength (nm)', fontsize=22, labelpad=15)
        ax.set_ylabel('Scale', fontsize=22, labelpad=10)
        ax.set_zlabel('Wavelet Power', fontsize=22, labelpad=10)

        # 设置坐标轴刻度
        ax.set_xlim(350, 2500)
        ax.set_ylim(3, 6)
        ax.set_xticks([500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500])
        ax.set_yticks([4, 5])
        ax.set_zticks([-1, -0.5, 0, 0.5, 1])
        ax.set_zlim(-1.5, 1.5)

        # 设置刻度粗细和坐标轴的粗细
        ax.tick_params(axis='both', which='major', width=2, labelsize=14)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)

        plt.show()

if __name__ == "__main__":
    filepath = input("请输入反射率Excel文件路径：")
    wavelet_transform = WaveletTransform()
    df_WF4, df_WF5 = wavelet_transform.transform(pd.read_excel(filepath))
    wavelet_transform.visualize(df_WF4, df_WF5)
