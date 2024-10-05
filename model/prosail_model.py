import prosail
import numpy as np
import itertools
import pandas as pd

class CartesianProduct:
    def __init__(self, *args):
        self.data = list(args)

    def build(self):
        return list(itertools.product(*self.data))

class ProsailModel:
    def __init__(self, N, Cab, Car, Cb, Cw, Cm, LAI, lidfa, hspot, tts, tto, psi, psoil1):
        self.N = N
        self.Cab = Cab
        self.Car = Car
        self.Cb = Cb
        self.Cw = Cw
        self.Cm = Cm
        self.LAI = LAI
        self.lidfa = lidfa
        self.hspot = hspot
        self.tts = tts
        self.tto = tto
        self.psi = psi
        self.psoil1 = psoil1
        self.parameters = self.generate_parameters()

    def generate_parameters(self):
        cartesian_product = CartesianProduct(self.N, self.Cab, self.Car, self.Cb, self.Cw, self.Cm, self.LAI,
                                             self.lidfa, self.hspot, self.tts, self.tto, self.psi, self.psoil1)
        return np.array(cartesian_product.build())

    def run_simulations(self):
        results = np.zeros([self.parameters.shape, 2101 + 1], dtype=float)
        for i, params in enumerate(self.parameters):
            results[i, :2101] = prosail.run_prosail(*params, prospect_version="D", typelidf=2, lidfb=-0.15, rsoil=1,
                                                    psoil=params[-1], factor="SDR")
            results[i, 2101:] = [params[self.Cm_index]]
        return results

    def save_results(self, results, filename=".csv"):#替换成你自己的路径
        df_results = pd.DataFrame(results, columns=[f"Wavelength_{i}" for i in range(2101)] + ["Cm"])
        df_results.to_csv(filename, index=False)
        print(f"模拟结果已成功保存到{filename}")
