"""ROMAN neural-network Teukolsky amplitudes for Schwarzschild eccentric orbits.

:class:`RomanAmplitude` is the amplitude module of
:class:`~few.waveform.waveform.FastSchwarzschildEccentricFlux`.
"""

# Flux-based Schwarzschild Eccentric amplitude module for Fast EMRI Waveforms
# performs calculation with a Roman network

import h5py
import numpy as np
from scipy.interpolate import RectBivariateSpline

from few.utils.globals import get_logger

# Python imports
from ..utils.baseclasses import BackendLike, SchwarzschildEccentric
from ..utils.citations import REFERENCE
from ..utils.globals import get_file_manager
from ..utils.mappings.schwarzecc import schwarzecc_p_to_y
from .base import AmplitudeBase


class RomanAmplitude(AmplitudeBase, SchwarzschildEccentric):
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
    coefficients for the modes in the reduced basis (input ``(y(p, e), e)``, see
    :func:`~few.utils.mappings.schwarzecc.schwarzecc_p_to_y`) and transforms it back to
    amplitude space; by default the amplitudes are then renormalized to the bicubic-spline
    flux norm of ``AmplitudeVectorNorm.dat``. Data files: ``SchwarzschildEccentricInput.hdf5``
    (weights, biases, reduced basis) and ``AmplitudeVectorNorm.dat``, fetched through the
    file manager. The constructor resets the inherited ``num_teuk_modes`` to the
    ``num_teuk_modes`` attribute of ``SchwarzschildEccentricInput.hdf5``.

    This module is available for GPU and CPU (compiled ``neural_layer_wrap`` and
    ``transform_output_wrap`` of the backend's ``pymatmul``).

    Args:
        buffer_length (int): Number of trajectory points the work buffers are allocated for.
            A longer input logs a warning and reallocates the buffers. Default ``1000``.
        force_backend: Backend name or object (see
            :class:`~few.utils.baseclasses.FewBackendConsumer`). Default ``None``.
        **kwargs: Forwarded to :class:`~few.utils.baseclasses.SchwarzschildEccentric`
            (``lmax`` default 10, ``nmax`` default 30).

    Attributes:
        data_file (str): Network data file name (``"SchwarzschildEccentricInput.hdf5"``).
        break_index (int): Half the network output length (number of real/imaginary pairs).
        transform_factor_inv (float): Inverse of the file's scalar transform factor.
        buffer_length (int): Current buffer length (number of points).
        weights (list[xp.ndarray]): Weight matrix of each layer, shape ``(dim1, dim2)``.
        bias (list[xp.ndarray]): Bias of each layer, shape ``(dim2,)``.
        dim1 (list[int]): First dimension of each weight matrix.
        dim2 (list[int]): Second dimension of each weight matrix.
        num_layers (int): Number of network layers.
        transform_matrix (xp.ndarray): Complex reduced-basis matrix mapping the network
            output to the amplitudes.
        max_num (int): Largest weight-matrix dimension (sets the buffer size).
        temp_mats (list[xp.ndarray]): The two flat work buffers the layers alternate between.
        run_relu_arr (np.ndarray): Per layer, 1 if the ReLU activation runs (0 for the last).
        amp_norm_spline (scipy.interpolate.RectBivariateSpline): Amplitude norm over
            ``(y, e)``.

    Example:
        .. code-block:: python

            import numpy as np
            from few.amplitude.romannet import RomanAmplitude

            amp = RomanAmplitude(force_backend="cpu")
            # a = 0, p [M], e, xI = 1; returns shape (2, amp.num_teuk_modes)
            A = amp(0.0, np.array([10.0, 11.0]), np.array([0.2, 0.2]), 1.0)
    """

    # TODO check: keep/remove/document -- doc-only stub (body is pass), never called
    def attributes_RomanAmplitude(self):
        """
        attributes:
            break_index (int): length of output vector from network divded by 2.
                It is really the number of pairs of real and imaginary numbers.
            neural_layer (obj): C++ class for computing neural network operations
            transform_output(obj): C++ class for transforming output from
                neural network in the reduced basis to the full amplitude basis.
            num_teuk_modes (int): number of teukolsky modes in the data file.
            transform_factor_inv (double): Inverse of the scalar transform factor.
                For this model, that is 1000.0.
            buffer_length (int): This class uses buffers. This is the maximum length
                the user expects for the input arrays.
            weights (list of xp.ndarrays): List of the weight matrices for each
                layer of the neural network. They are flattened for entry into
                C++ in column-major order. They have shape (dim1, dim2).
            bias (list of xp.ndarrays): List of the bias arrays for each layer
                of the neural network. They have shape (dim2,).
            dim1 (list of int): List of 1st dimension length in each layer.
            dim2 (list of int): List of 2nd dimension length in each layer.
            num_layers (int): Number of layers in the neural network.
            transform_matrix (2D complex128 xp.ndarray): Matrix for tranforming
                output of neural network onto original amplitude basis.
            max_num (int): Figures out the maximum dimension of all weight matrices
                for buffers.
            temp_mats (len-2 list of double xp.ndarrays): List that holds
                temporary matrices for neural network evaluation. Each layer switches
                between which is the input and output to properly interface with
                cBLAS/cuBLAS.
            run_relu_arr  (1D int xp.ndarray): Array holding information about
                whether each layer will run the relu activation. All layers have
                value 1, except for the last layer with value 0.

        """
        pass

    def __init__(self, buffer_length=1000, force_backend: BackendLike = None, **kwargs):
        AmplitudeBase.__init__(self)
        SchwarzschildEccentric.__init__(self, **kwargs, force_backend=force_backend)

        # check if user has the necessary data
        # if not, the data will automatically download
        self.data_file = "SchwarzschildEccentricInput.hdf5"

        # get information about this specific model from the file
        with h5py.File(get_file_manager().get_file(self.data_file), "r") as fp:
            num_teuk_modes = fp.attrs["num_teuk_modes"]
            transform_factor = fp.attrs["transform_factor"]
            self.break_index = fp.attrs["break_index"]

        self.num_teuk_modes = num_teuk_modes
        self.transform_factor_inv = 1 / transform_factor

        self.buffer_length = buffer_length

        self._initialize_weights()

        # setup amplitude normalization
        y_in, e_in, norm = np.genfromtxt(
            get_file_manager().get_file("AmplitudeVectorNorm.dat")
        ).T

        num_y = len(np.unique(y_in))
        num_e = len(np.unique(e_in))

        self.amp_norm_spline = RectBivariateSpline(
            np.unique(y_in), np.unique(e_in), norm.reshape(num_e, num_y).T
        )

    @property
    def neural_layer(self):
        """The backend's compiled dense-layer kernel (``neural_layer_wrap``)."""
        return self.backend.neural_layer_wrap

    @property
    def transform_output(self):
        """The backend's compiled reduced-basis-to-amplitude kernel (``transform_output_wrap``)."""
        return self.backend.transform_output_wrap

    @classmethod
    def module_references(cls) -> list[REFERENCE]:
        """Return citations related to this module (``ROMANNET`` plus the parents')."""
        return [REFERENCE.ROMANNET] + super().module_references()

    @classmethod
    def supported_backends(cls):
        """Backend preference order: ``["cuda13x", "cuda12x", "cuda11x", "cpu"]``."""
        return cls.GPU_RECOMMENDED()

    def _initialize_weights(self):
        """Load the layer weights, biases and reduced basis from the data file; allocate buffers."""
        # initalize weights/bias/dimensions for the neural network

        self.weights = []
        self.bias = []
        self.dim1 = []
        self.dim2 = []

        # get highest layer number
        self.num_layers = 0

        # extract all necessary information from the file
        with h5py.File(get_file_manager().get_file(self.data_file), "r") as fp:
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

            # get the post network transform matrix
            self.transform_matrix = self.xp.asarray(fp["reduced_basis"])

        # longest length in any dimension for buffers
        self.max_num = np.max([self.dim1, self.dim2])

        # declare buffers
        # each layer will alternate between these arrays as the input and output
        # of the layer. The input is multiplied by the layer weight.
        self.temp_mats = [
            self.xp.zeros((self.max_num * self.buffer_length,), dtype=self.xp.float64),
            self.xp.zeros((self.max_num * self.buffer_length,), dtype=self.xp.float64),
        ]

        # array for letting C++ know if the layer is activated
        self.run_relu_arr = np.ones(self.num_layers, dtype=int)
        self.run_relu_arr[-1] = 0

    def get_amplitudes(self, a, p, e, xI, *args, specific_modes=None, renormalize_amps=True, **kwargs):
        """Evaluate the ROMAN network along a trajectory (called by ``__call__``).

        This function takes the trajectory in :math:`(p,e)` as arrays and returns the complex
        amplitude of the modes to adiabatic order at each step of the trajectory.

        Args:
            a (xp.ndarray): Spin; not used (Schwarzschild).
            p (xp.ndarray): Semi-latus rectum in units of :math:`M`, shape ``(N,)``.
            e (xp.ndarray): Eccentricity, shape ``(N,)``.
            xI (xp.ndarray): :math:`x_I`; not used.
            *args: Accepted and ignored.
            specific_modes (xp.ndarray, optional): Mode indices (columns) to return. Default
                ``None`` means all modes.
            renormalize_amps (bool, optional): If ``True`` (default), rescale the amplitudes at
                each point so that their total norm (including the :math:`m < 0` modes) matches
                the bicubic-spline norm ``amp_norm_spline``; this slightly improves accuracy.
            **kwargs: Accepted and ignored.

        Returns:
            xp.ndarray: Complex amplitudes, shape ``(N, num_teuk_modes)``, or
            ``(N, len(specific_modes))``.
        """

        input_len = len(p)

        # check ifn input_len is greater than the buffer_length attribute
        # if so reset the buffers and update the attribute
        if input_len > self.buffer_length:
            get_logger().warning(
                "Input length %u is larger than initial buffer_length (%u). Reallocating preallocated arrays for this size.",
                input_len,
                self.buffer_length,
            )
            self.buffer_length = input_len

            self.temp_mats = [
                self.xp.zeros(
                    (self.max_num * self.buffer_length,), dtype=self.xp.float64
                ),
                self.xp.zeros(
                    (self.max_num * self.buffer_length,), dtype=self.xp.float64
                ),
            ]

        # the input is (y, e)
        y = schwarzecc_p_to_y(p, e, use_gpu=self.backend.uses_cupy)

        # column-major single dimensional array input
        input = self.xp.concatenate([y, e])

        # fill first temporary matrix
        self.temp_mats[0][: 2 * input_len] = input

        # setup arrays
        # teukolsky mode (final output)
        teuk_modes = self.xp.zeros(
            (input_len * self.num_teuk_modes,), dtype=self.xp.complex128
        )

        # neural network output
        nn_out_mat = self.xp.zeros(
            (input_len * self.break_index,), dtype=self.xp.complex128
        )

        # run the neural network
        for i, (weight, bias, run_relu) in enumerate(
            zip(self.weights, self.bias, self.run_relu_arr)
        ):
            # set temporary input and output matrix
            mat_in = self.temp_mats[i % 2]
            mat_out = self.temp_mats[(i + 1) % 2]

            # get shape information
            m = len(p)
            k, n = weight.shape

            # run the C++ neural net layer
            self.neural_layer(
                mat_out, mat_in, weight.T.flatten(), bias, m, k, n, run_relu
            )

        # transform the neural net ouput back to the amplitude space
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

        # reshape the teukolsky modes
        teuk_modes = teuk_modes.reshape(self.num_teuk_modes, input_len).T

        # norm with respect to the flux interpolated by the bicubic spline
        if renormalize_amps:
            amp_norm = self.amp_norm_spline.ev(y, e)
            amp_for_norm = self.xp.sum(
                self.xp.abs(
                    self.xp.concatenate(
                        [teuk_modes, self.xp.conj(teuk_modes[:, self.m0mask])],
                        axis=1,
                    )
                )
                ** 2,
                axis=1,
            ) ** (1 / 2)

            # normalize
            factor = amp_norm / amp_for_norm
            teuk_modes = teuk_modes * factor[:, None]

        if specific_modes is not None:
            return teuk_modes[:, specific_modes]
        else:
            return teuk_modes
