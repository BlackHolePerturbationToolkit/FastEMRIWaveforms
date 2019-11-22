import os
import time

try:
    import cupy as xp
except ImportError:
    import numpy as xp

import numpy as np


class LeakyReLU:
    def __init__(self, alpha=0.2):
        self.alpha = 0.2

    def __call__(self, x):
        return x * (x >= 0.0) + 0.2 * x * (x < 0.0)


class Linear:
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def __call__(self, x):
        return xp.matmul(x, self.w) + self.b


class NN:
    def __init__(self, input_str="SE_", folder="weights/", activation_kwargs={}):
        self.layers = []
        file_list = os.listdir(folder)

        # get highest layer number
        self.num_layers = 0
        for fp in file_list:
            layer_num = int(fp.split("SE_")[1][1:].split(".")[0])
            if layer_num > self.num_layers:
                self.num_layers = layer_num

        for i in range(1, self.num_layers + 1):
            temp = {}
            for let in ["w", "b"]:
                mat = np.genfromtxt(folder + input_str + let + str(i) + ".txt")
                temp[let] = xp.asarray(mat, dtype=xp.float32)

            self.layers.append(Linear(temp["w"], temp["b"]))

        self.activation = LeakyReLU(**activation_kwargs)

        print("Number of layers:", self.num_layers)

    def __call__(self, x_in):
        x = x_in.copy()
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_layers - 1:
                x = self.activation(x)
        return x


if __name__ == "__main__":
    import pdb

    num_test = 126757
    e = xp.random.uniform(0.1, 0.5, num_test, dtype=xp.float32)
    p = xp.random.uniform(7.0, 11.0, num_test, dtype=xp.float32)

    test = xp.asarray([p, e], dtype=xp.float32).T
    check = NN()
    import time

    for j in range(100):
        st = time.perf_counter()
        for i in range(10):
            out = check(test)
        et = time.perf_counter()
        print((et - st) / 10)
    pdb.set_trace()
