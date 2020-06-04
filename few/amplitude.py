import numpy as np
import os

from pymatmul import neural_layer_wrap, transform_output_wrap


try:
    import cupy as xp

except ImportError:
    import numpy as xp

RUN_RELU = 1
NO_RELU = 0


class Amplitude:
    def __init__(
        self,
        input_str="SE_n30_double_",
        folder="few/files/weights/",
        activation_kwargs={},
        num_teuk_modes=3843,
        transform_factor=1000.0,
        transform_file="few/files/reduced_basis_n30_new.dat",
        max_input_len=1000,
    ):
        self.num_teuk_modes = num_teuk_modes
        self.transform_factor_inv = 1 / transform_factor
        self.transform_file = transform_file

        self.transform_matrix = xp.asarray(
            np.genfromtxt(transform_file, dtype=xp.complex128)
        ).flatten()
        self.break_index = 97
        self.max_input_len = max_input_len

        self._initialize_weights(input_str=input_str, folder=folder)

    def _initialize_weights(
        self, input_str="SE_n30_double_", folder="few/files/weights/"
    ):
        self.weights = []
        self.bias = []
        self.dim1 = []
        self.dim2 = []

        file_list = os.listdir(folder)

        # get highest layer number
        self.num_layers = 0
        for fp in file_list:
            layer_num = int(fp.split(input_str)[1][1:].split(".")[0])
            if layer_num > self.num_layers:
                self.num_layers = layer_num

        for i in range(1, self.num_layers + 1):
            temp = {}
            for let in ["w", "b"]:
                mat = np.genfromtxt(folder + input_str + let + str(i) + ".txt")
                temp[let] = xp.asarray(mat)

            self.weights.append(temp["w"])
            self.bias.append(temp["b"])
            self.dim1.append(temp["w"].shape[0])
            self.dim2.append(temp["w"].shape[1])

        self.max_num = np.max([self.dim1, self.dim2])

        self.temp_mats = [
            xp.zeros((self.max_num * self.max_input_len,), dtype=xp.float64),
            xp.zeros((self.max_num * self.max_input_len,), dtype=xp.float64),
        ]
        self.run_relu_arr = np.ones(self.num_layers, dtype=int)
        self.run_relu_arr[-1] = 0

    def __call__(self, p, e):
        input_len = len(p)
        input = xp.concatenate([p, e])
        self.temp_mats[0][: 2 * input_len] = input

        teuk_modes = xp.zeros((input_len * self.num_teuk_modes,), dtype=xp.complex128)
        nn_out_mat = xp.zeros((input_len * self.break_index))

        for i, (weight, bias, run_relu) in enumerate(
            zip(self.weights, self.bias, self.run_relu_arr)
        ):

            mat_in = self.temp_mats[i % 2]
            mat_out = self.temp_mats[(i + 1) % 2]

            m = len(p)
            k, n = weight.shape

            neural_layer_wrap(
                mat_out, mat_in, weight.flatten(), bias, m, k, n, run_relu
            )

        transform_output_wrap(
            teuk_modes,
            self.transform_matrix,
            nn_out_mat,
            mat_out,
            input_len,
            self.break_index,
            self.transform_factor_inv,
            self.num_teuk_modes,
        )
        return teuk_modes.reshape(input_len, self.num_teuk_modes)
