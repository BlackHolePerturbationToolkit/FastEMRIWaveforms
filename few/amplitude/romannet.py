import numpy as np
import os
import h5py
import warnings

dir_path = os.path.dirname(os.path.realpath(__file__))

from pymatmul_cpu import neural_layer_wrap as neural_layer_wrap_cpu
from pymatmul_cpu import transform_output_wrap as transform_output_wrap_cpu

import pymatmul_cpu

from few.utils.baseclasses import SchwarzschildEccentric

try:
    import cupy as xp
    from pymatmul import neural_layer_wrap, transform_output_wrap

    run_gpu = True

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp

    run_gpu = False

RUN_RELU = 1
NO_RELU = 0


class ROMANAmplitude(SchwarzschildEccentric):
    """Calculate Teukolsky amplitudes with a ROMAN.

    ROMAN stands for reduced-order models with artificial neurons. Please see
    the documentations for
    :class:`few.utils.baseclasses.SchwarzschildEccentric`
    for overall aspects of these models.

    A reduced order model is computed for :math:`A_{lmn}`. The data sets that
    are provided over a grid of :math:`(p,e)` were provided by Scott Hughes.

    A feed-foward neural network is then trained on the ROM. Its weights are
    used in this module.

    When the user inputs :math:`(p,e)`, the neural network determines
    coefficients for the modes in the reduced basic and transforms it back to
    amplitude space.

    This module is available for GPU and CPU.


    args:
        max_input_len (int, optional): Number of points to initialize for
            buffers. This allows the user to limit memory usage. However, if the
            user requests more length, a warning will be thrown and the
            max_input_len will be increased accordingly and arrays reallocated.
            Default is 1000.

        **kwargs (dict, optional): Keyword arguments for the base class:
            :class:`few.utils.baseclasses.SchwarzschildEccentric`. Default is
            {}.

    """

    def __init__(self, max_input_len=1000, **kwargs):

        SchwarzschildEccentric.__init__(self, **kwargs)

        self.folder = dir_path + "/../files/"
        self.data_file = "SchwarzschildEccentricInput.hdf5"

        with h5py.File(self.folder + self.data_file, "r") as fp:
            num_teuk_modes = fp.attrs["num_teuk_modes"]
            transform_factor = fp.attrs["transform_factor"]
            self.break_index = fp.attrs["break_index"]

        if self.use_gpu:
            self.neural_layer = neural_layer_wrap
            self.transform_output = transform_output_wrap

        else:
            self.neural_layer = neural_layer_wrap_cpu
            self.transform_output = transform_output_wrap_cpu

        self.num_teuk_modes = num_teuk_modes
        self.transform_factor_inv = 1 / transform_factor

        self.max_input_len = max_input_len

        self._initialize_weights()

    @property
    def citation(self):
        return """
                @article{Chua:2018woh,
                    author = "Chua, Alvin J.K. and Galley, Chad R. and Vallisneri, Michele",
                    title = "{Reduced-order modeling with artificial neurons for gravitational-wave inference}",
                    eprint = "1811.05491",
                    archivePrefix = "arXiv",
                    primaryClass = "astro-ph.IM",
                    doi = "10.1103/PhysRevLett.122.211101",
                    journal = "Phys. Rev. Lett.",
                    volume = "122",
                    number = "21",
                    pages = "211101",
                    year = "2019"
                }
                """

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

    def __call__(self, p, e, *args, specific_modes=None, **kwargs):
        """Calculate Teukolsky amplitudes for Schwarzschild eccentric.

        This function takes the inputs the trajectory in :math:`(p,e)` as arrays
        and returns the complex amplitude of all modes to adiabatic order at
        each step of the trajectory.

        args:
            p (1D double numpy.ndarray): Array containing the trajectory for values of
                the semi-latus rectum.
            e (1D double numpy.ndarray): Array containing the trajectory for values of
                the eccentricity.
            *args (tuple, placeholder): Added to create flexibility when calling different
                amplitude modules. It is not used.
            specific_modes (list, optional): List of tuples for (l, m, n) values
                desired modes. Default is None.
            **kwargs (dict, placeholder): Added to create flexibility when calling different
                amplitude modules. It is not used.

        returns:
            2D array (double): If specific_modes is None, Teukolsky modes in shape (number of trajectory points, number of modes)
            dict: Dictionary with requested modes.


        """
        input_len = len(p)

        if input_len > self.max_input_len:
            warnings.warn(
                "Input length {} is larger than initial max_input_len ({}). Reallocating preallocated arrays for this size.".format(
                    input_len, self.max_input_len
                )
            )
            self.max_input_len = input_len

            self.temp_mats = [
                self.xp.zeros(
                    (self.max_num * self.max_input_len,), dtype=self.xp.float64
                ),
                self.xp.zeros(
                    (self.max_num * self.max_input_len,), dtype=self.xp.float64
                ),
            ]

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

        teuk_modes = teuk_modes.reshape(self.num_teuk_modes, input_len).T
        if specific_modes is None:
            return teuk_modes

        else:
            number_of_modes_for_return = len(specific_modes)
            temp = {}
            for lmn in specific_modes:
                temp[lmn] = teuk_modes[:, self.special_index_map[lmn]]
                l, m, n = lmn
                if m < 0:
                    temp[lmn] = self.xp.conj(temp[lmn])

            return temp
