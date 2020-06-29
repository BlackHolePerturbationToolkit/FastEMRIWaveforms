import numpy as np
import os
import h5py

from pymatmul_cpu import neural_layer_wrap as neural_layer_wrap_cpu
from pymatmul_cpu import transform_output_wrap as transform_output_wrap_cpu
from pyInterp2DAmplitude import Interp2DAmplitude_wrap, pyAmplitudeCarrier

try:
    import cupy as xp
    from pymatmul import neural_layer_wrap, transform_output_wrap

    run_gpu = True

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp

    run_gpu = False

RUN_RELU = 1
NO_RELU = 0


class ROMANAmplitude:
    def __init__(self, activation_kwargs={}, max_input_len=1000, use_gpu=False):

        self.folder = "few/files/"
        self.data_file = "SchwarzschildEccentricInput.hdf5"

        with h5py.File(self.folder + self.data_file, "r") as fp:
            num_teuk_modes = fp.attrs["num_teuk_modes"]
            transform_factor = fp.attrs["transform_factor"]
            self.break_index = fp.attrs["break_index"]

        self.use_gpu = use_gpu
        if use_gpu:
            self.xp = xp
            self.neural_layer = neural_layer_wrap
            self.transform_output = transform_output_wrap

        else:
            self.xp = np
            self.neural_layer = neural_layer_wrap_cpu
            self.transform_output = transform_output_wrap_cpu

        self.num_teuk_modes = num_teuk_modes
        self.transform_factor_inv = 1 / transform_factor

        self.max_input_len = max_input_len

        self._initialize_weights()

    def _initialize_weights(self):
        self.weights = []
        self.bias = []
        self.dim1 = []
        self.dim2 = []

        # get highest layer number
        self.num_layers = 0
        with h5py.File(self.folder + self.data_file, "r") as fp:
            for key, value in fp.items():
                if key == "reduced_basis":
                    continue

                layer_num = int(key[1:])

                if layer_num > self.num_layers:
                    self.num_layers = layer_num

            for i in range(1, self.num_layers + 1):
                temp = {}
                for let in ["w", "b"]:
                    mat = fp.get(let + str(i))[:]
                    temp[let] = self.xp.asarray(mat)

                self.weights.append(temp["w"])
                self.bias.append(temp["b"])
                self.dim1.append(temp["w"].shape[0])
                self.dim2.append(temp["w"].shape[1])

            self.transform_matrix = self.xp.asarray(fp["reduced_basis"])

        self.max_num = np.max([self.dim1, self.dim2])

        self.temp_mats = [
            self.xp.zeros((self.max_num * self.max_input_len,), dtype=self.xp.float64),
            self.xp.zeros((self.max_num * self.max_input_len,), dtype=self.xp.float64),
        ]
        self.run_relu_arr = np.ones(self.num_layers, dtype=int)
        self.run_relu_arr[-1] = 0

    def _p_to_y(self, p, e):

        return self.xp.log(-(21 / 10) - 2 * e + p)

    def __call__(self, p, e, *args):
        input_len = len(p)

        y = self._p_to_y(p, e)
        input = self.xp.concatenate([y, e])
        self.temp_mats[0][: 2 * input_len] = input

        teuk_modes = self.xp.zeros(
            (input_len * self.num_teuk_modes,), dtype=self.xp.complex128
        )
        nn_out_mat = self.xp.zeros(
            (input_len * self.break_index,), dtype=self.xp.complex128
        )

        for i, (weight, bias, run_relu) in enumerate(
            zip(self.weights, self.bias, self.run_relu_arr)
        ):

            mat_in = self.temp_mats[i % 2]
            mat_out = self.temp_mats[(i + 1) % 2]

            m = len(p)
            k, n = weight.shape

            self.neural_layer(
                mat_out, mat_in, weight.T.flatten(), bias, m, k, n, run_relu
            )

        self.transform_output(
            teuk_modes,
            self.transform_matrix.T.flatten(),
            nn_out_mat,
            mat_out,
            input_len,
            self.break_index,
            self.transform_factor_inv,
            self.num_teuk_modes,
        )

        return teuk_modes.reshape(self.num_teuk_modes, input_len).T


class Interp2DAmplitude:
    def __init__(self, num_teuk_modes=3843, lmax=10, nmax=30):

        self.amplitude_carrier = pyAmplitudeCarrier(lmax, nmax)
        self.num_modes = num_teuk_modes

    def __call__(self, p, e, l_arr, m_arr, n_arr):

        input_len = len(p)
        teuk_modes = Interp2DAmplitude_wrap(
            p,
            e,
            l_arr.astype(np.int32),
            m_arr.astype(np.int32),
            n_arr.astype(np.int32),
            input_len,
            self.num_modes,
            self.amplitude_carrier,
        )
        return teuk_modes
