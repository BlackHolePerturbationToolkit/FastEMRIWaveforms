# Flux-based Schwarzschild Eccentric amplitude module for Fast EMRI Waveforms
# performs calculation with a Roman network

# Copyright (C) 2020 Michael L. Katz, Alvin J.K. Chua, Niels Warburton, Scott A. Hughes
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import warnings

import numpy as np
import h5py

# Cython/C++ imports
#from pycpupnamp import Zlmkn8_5PNe10 as Zlmkn8_5PNe10_cpu

# Python imports
from few.utils.baseclasses import Pn5AdiabaticAmp, AmplitudeBase, ParallelModuleBase
from few.utils.utility import get_fundamental_frequencies, Y_to_xI
from few.utils.citations import *

# check for cupy and GPU version of pymatmul
try:
    # Cython/C++ imports
    from pypnamp import Zlmkn8_5PNe10, Zlmkn8_5PNe10

    # Python imports
    import cupy as xp

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp


# get path to this file
dir_path = os.path.dirname(os.path.realpath(__file__))


class Pn5Amplitude(AmplitudeBase, Pn5AdiabaticAmp, ParallelModuleBase):
    """Calculate Teukolsky amplitudes with a ROMAN.

    # TODO: update
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
        max_init_len (int, optional): Number of points to initialize for
            buffers. This allows the user to limit memory usage. However, if the
            user requests more length, a warning will be thrown and the
            max_init_len will be increased accordingly and arrays reallocated.
            Default is 1000.
        **kwargs (dict, optional): Keyword arguments for the base classes:
            :class:`few.utils.baseclasses.SchwarzschildEccentric`,
            :class:`few.utils.baseclasses.AmplitudeBase`,
            :class:`few.utils.baseclasses.ParallelModuleBase`.
            Default is {}.

    """

    def attributes_Pn5Amplitude(self):
        """
        attributes:
            few_dir (str): absolute path to the FastEMRIWaveforms directory
            break_index (int): length of output vector from network divded by 2.
                It is really the number of pairs of real and imaginary numbers.
            use_gpu (bool): If True, use the GPU.
            neural_layer (obj): C++ class for computing neural network operations
            transform_output(obj): C++ class for transforming output from
                neural network in the reduced basis to the full amplitude basis.
            num_teuk_modes (int): number of teukolsky modes in the data file.
            transform_factor_inv (double): Inverse of the scalar transform factor.
                For this model, that is 1000.0.
            max_init_len (int): This class uses buffers. This is the maximum length
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

    def __init__(self, max_init_len=1000, **kwargs):

        ParallelModuleBase.__init__(self, **kwargs)
        Pn5AdiabaticAmp.__init__(self, **kwargs)
        AmplitudeBase.__init__(self, **kwargs)

        # adjust c++ classes based on gpu usage
        if self.use_gpu:
            self.Zlmkn8_5PNe10 = Zlmkn8_5PNe10

        #else:
            #self.Zlmkn8_5PNe10 = Zlmkn8_5PNe10_cpu

    @property
    def citation(self):
        """Return citations for this module"""
        return romannet_citation + larger_few_citation + few_citation + few_software_citation

    @property
    def gpu_capability(self):
        """Confirms GPU capability"""
        return True

    def get_amplitudes(self, q, p, e, Y, theta, *args, specific_modes=None, include_spheroidal_harmonics=True, **kwargs):
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
        
        if specific_modes is not None:
            if not isinstance(specific_modes, list):
                raise ValueError("If providing specific_modes, needs to be a list of tuples.")
            
            modes_tmp = self.xp.asarray(specific_modes)
            if modes_tmp.shape[0] == 4 and modes_tmp.shape[1] != 4:
                l_all, m_all, k_all, n_all = [modes_tmp[i].copy() for i in range(4)]
                num_modes = modes_tmp.shape[1]
                
            elif modes_tmp.shape[1] == 4 and modes_tmp.shape[0] != 4:
                l_all, m_all, k_all, n_all = [modes_tmp.T[i].copy() for i in range(4)]
                num_modes = modes_tmp.shape[0]

            else:
                if modes_tmp.shape[1] == 4 and modes_tmp.shape[0] == 4:
                    raise ValueError("Input specific_modes shape is (4, 4), which is not allowed. A length of 4 is reserved for the dimension along lmkn.")
                else:
                    raise ValueError(f"specific_modes has shape {modes_tmp.shape}. One dimension must be 4 indicating lmkn.")

        else:
            num_modes = self.num_teuk_modes
            l_all, m_all, k_all, n_all = self.l_arr.copy(), self.m_arr.copy(), self.k_arr.copy(), self.n_arr.copy()
    
    
        l_all, m_all, k_all, n_all = l_all.astype(self.xp.int32), m_all.astype(self.xp.int32), k_all.astype(self.xp.int32), n_all.astype(self.xp.int32)

        
        # 14 ms single thread
        # convert Y to x_I for fund freqs
        try:
            q_in = q.get()
        except AttributeError:
            q_in = q

        try:
            p_in = p.get()
            e_in = e.get()
            Y_in = Y.get()
        except AttributeError:
            p_in = p
            e_in = e
            Y_in = Y

        xI = Y_to_xI(q_in, p_in, e_in, Y_in)

        # 13 ms single thread
        # these are dimensionless and in radians
        OmegaPhi, OmegaTheta, OmegaR = get_fundamental_frequencies(q_in, p_in, e_in, xI)

        # dimensionalize the frequencies
        # TODO: do I need to dimensionalize
        #OmegaPhi, OmegaTheta, OmegaR = (
        #    OmegaPhi / Msec,
        #    OmegaTheta / Msec,
        #    OmegaR / Msec,
        #)
        Almkn_out = self.xp.zeros((2 * num_modes * input_len,), dtype=complex)

        p_in, e_in, Y_in, q_in, theta_in, OmegaPhi_in, OmegaTheta_in, OmegaR_in = self.xp.asarray(self.xp.atleast_1d(p)), self.xp.asarray(self.xp.atleast_1d(e)), self.xp.asarray(self.xp.atleast_1d(Y)), self.xp.asarray(self.xp.atleast_1d(q)), self.xp.asarray(self.xp.atleast_1d(theta)), self.xp.asarray(self.xp.atleast_1d(OmegaPhi)), self.xp.asarray(self.xp.atleast_1d(OmegaTheta)), self.xp.asarray(self.xp.atleast_1d(OmegaR))

        if len(p_in) != len(theta_in):
            if len(theta_in) != 1:
                raise ValueError("Length of theta array must be same as p,e,Y if input as an array rather than a scalar.")
            theta_in = self.xp.repeat(theta_in, len(p_in))

        if len(p_in) != len(q_in):
            if len(q_in) != 1:
                raise ValueError("Length of theta array must be same as p,e,Y if input as an array rather than a scalar.")
            q_in = self.xp.repeat(q_in, len(p_in))

        self.Zlmkn8_5PNe10(Almkn_out, l_all, m_all, k_all, n_all, q_in, theta_in, p_in, e_in, Y_in, OmegaR_in, OmegaTheta_in, OmegaPhi_in, num_modes, input_len, include_spheroidal_harmonics)

        # reshape the teukolsky modes
        # len(dim0) = 2, 0 is right, 1 is left
        Almkn_out = Almkn_out.reshape(2, num_modes, input_len).T

        return Almkn_out
