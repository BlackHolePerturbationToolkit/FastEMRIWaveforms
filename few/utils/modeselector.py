# Online mode selection for FastEMRIWaveforms Packages

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


import numpy as np

import os

from few.utils.citations import *
from few.utils.utility import get_fundamental_frequencies
from few.utils.constants import *
from few.utils.baseclasses import ParallelModuleBase

# check for cupy
try:
    import cupy as cp

except (ImportError, ModuleNotFoundError) as e:
    import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))

class ModeSelector(ParallelModuleBase):
    """Filter teukolsky amplitudes based on power contribution.

    This module takes teukolsky modes, combines them with their associated ylms,
    and determines the power contribution from each mode. It then filters the
    modes bases on the fractional accuracy on the total power (eps) parameter.
    Additionally, if a sensitivity curve is provided, the mode power is also
    weighted according to the PSD of the sensitivity.

    The mode filtering is a major contributing factor to the speed of these
    waveforms as it removes large numbers of useles modes from the final
    summation calculation.

    Be careful as this is built based on the construction that input mode arrays
    will in order of :math:`m=0`, :math:`m>0`, and then :math:`m<0`.

    args:
        l_arr (1D int xp.ndarray): The l-mode indices for each mode index.
        m_arr (1D int xp.ndarray): The m-mode indices for each mode index.
        n_arr (1D int xp.ndarray): The n-mode indices for each mode index.
        sensitivity_fn (object, optional): Sensitivity curve function that takes
            a frequency (Hz) array as input and returns the Power Spectral Density (PSD)
            of the sensitivity curve. Default is None. If this is not none, this
            sennsitivity is used to weight the mode values when determining which
            modes to keep. **Note**: if the sensitivity function is provided,
            and GPUs are used, then this function must accept CuPy arrays as input.
        **kwargs (dict, optional): Keyword arguments for the base classes:
            :class:`few.utils.baseclasses.ParallelModuleBase`.
            Default is {}.

    """

    def __init__(self, l_arr, m_arr, n_arr, sensitivity_fn=None, **kwargs):
        ParallelModuleBase.__init__(self, **kwargs)

        if self.use_gpu:
            xp = cp
        else:
            xp = np

        # store information releated to m values
        # the order is m = 0, m > 0, m < 0
        m0mask = m_arr != 0
        self.m0mask = m0mask
        self.num_m_zero_up = len(m0mask)
        self.num_m_1_up = len(xp.arange(len(m0mask))[m0mask])
        self.num_m0 = len(xp.arange(len(m0mask))[~m0mask])

        self.sensitivity_fn = sensitivity_fn

    @property
    def gpu_capability(self):
        """Confirms GPU capability"""
        return True
    
    @property
    def is_predictive(self):
        """Whether this mode selector should be used before or after amplitude generation"""
        return False
    
    def attributes_ModeSelector(self):
        """
        attributes:
            xp: cupy or numpy depending on GPU usage.
            num_m_zero_up (int): Number of modes with :math:`m\geq0`.
            num_m_1_up (int): Number of modes with :math:`m\geq1`.
            num_m0 (int): Number of modes with :math:`m=0`.
            sensitivity_fn (object): sensitivity generating function for power-weighting.

        """
        pass

    @property
    def citation(self):
        """Return citations related to this class."""
        return larger_few_citation + few_citation + few_software_citation

    def __call__(self, teuk_modes, ylms, modeinds, fund_freq_args=None, eps=1e-5):
        """Call to sort and filer teukolsky modes.

        This is the call function that takes the teukolsky modes, ylms,
        mode indices and fractional accuracy of the total power and returns
        filtered teukolsky modes and ylms.

        args:
            teuk_modes (2D complex128 xp.ndarray): Complex teukolsky amplitudes
                from the amplitude modules.
                Shape: (number of trajectory points, number of modes).
            ylms (1D complex128 xp.ndarray): Array of ylm values for each mode,
                including m<0. Shape is (num of m==0,) + (num of m>0,)
                + (num of m<0). Number of m<0 and m>0 is the same, but they are
                ordered as (m==0) first then m>0 then m<0.
            modeinds (list of int xp.ndarrays): List containing the mode index arrays. If in an
                equatorial model, need :math:`(l,m,n)` arrays. If generic,
                :math:`(l,m,k,n)` arrays. e.g. [l_arr, m_arr, n_arr].
            fund_freq_args (tuple, optional): Args necessary to determine
                fundamental frequencies along trajectory. The tuple will represent
                :math:`(M, a, e, p, \cos\iota)` where the large black hole mass (:math:`M`)
                and spin (:math:`a`) are scalar and the other three quantities are xp.ndarrays.
                This must be provided if sensitivity weighting is used. Default is None.
            eps (double, optional): Fractional accuracy of the total power used
                to determine the contributing modes. Lowering this value will
                calculate more modes slower the waveform down, but generally
                improving accuracy. Increasing this value removes modes from
                consideration and can have a considerable affect on the speed of
                the waveform, albeit at the cost of some accuracy (usually an
                acceptable loss). Default that gives good mismatch qualities is
                1e-5.

        """

        if self.use_gpu:
            xp = cp
        else:
            xp = np

        # get the power contribution of each mode including m < 0
        power = (
            xp.abs(
                xp.concatenate(
                    [teuk_modes, xp.conj(teuk_modes[:, self.m0mask])], axis=1
                )
                * ylms
            )
            ** 2
        )

        # if noise weighting
        if self.sensitivity_fn is not None:
            if fund_freq_args is None:
                raise ValueError(
                    "If sensitivity weighting is desired, the fund_freq_args kwarg must be provided."
                )

            M = fund_freq_args[0]
            Msec = M * MTSUN_SI

            # get dimensionless fundamental frequency
            OmegaPhi, OmegaTheta, OmegaR = get_fundamental_frequencies(
                *fund_freq_args[1:]
            )

            # get frequencies in Hz
            f_Phi, f_omega, f_r = OmegaPhi, OmegaTheta, OmegaR = (
                xp.asarray(OmegaPhi) / (Msec * 2 * PI),
                xp.asarray(OmegaTheta) / (Msec * 2 * PI),
                xp.asarray(OmegaR) / (Msec * 2 * PI),
            )

            # TODO: update when in kerr
            freqs = (
                modeinds[1][xp.newaxis, :] * f_Phi[:, xp.newaxis]
                + modeinds[2][xp.newaxis, :] * f_r[:, xp.newaxis]
            )

            freqs_shape = freqs.shape

            # make all frequencies positive
            freqs_in = xp.abs(freqs)
            PSD = self.sensitivity_fn(freqs_in.flatten()).reshape(freqs_shape)

            # weight by PSD
            power /= PSD

        # sort the power for a cumulative summation
        inds_sort = xp.argsort(power, axis=1)[:, ::-1]
        power = xp.sort(power, axis=1)[:, ::-1]
        cumsum = xp.cumsum(power, axis=1)

        # initialize and indices array for keeping modes
        inds_keep = xp.full(cumsum.shape, True)

        # keep modes that add to within the fractional power (1 - eps)
        inds_keep[:, 1:] = cumsum[:, :-1] < cumsum[:, -1][:, xp.newaxis] * (1 - eps)

        # finds indices of each mode to be kept
        temp = inds_sort[inds_keep]

        # adjust the index arrays to make -m indices equal to +m indices
        # if +m or -m contributes, we keep both because of structure of CUDA kernel
        temp = temp * (temp < self.num_m_zero_up) + (temp - self.num_m_1_up) * (
            temp >= self.num_m_zero_up
        )

        # if +m or -m contributes, we keep both because of structure of CUDA kernel
        keep_modes = xp.unique(temp)

        # set ylms

        # adust temp arrays specific to ylm setup
        temp2 = keep_modes * (keep_modes < self.num_m0) + (
            keep_modes + self.num_m_1_up
        ) * (keep_modes >= self.num_m0)

        # ylm duplicates the m = 0 unlike teuk_modes
        ylmkeep = xp.concatenate([keep_modes, temp2])

        # setup up teuk mode and ylm returns
        out1 = (teuk_modes[:, keep_modes], ylms[ylmkeep])

        # setup up mode values that have been kept
        out2 = tuple([arr[keep_modes] for arr in modeinds])

        return out1 + out2

class NeuralModeSelector(ParallelModuleBase):
    """Filter teukolsky amplitudes based on power contribution.

    This module uses a combination of a pre-computed mask and a feed-forward neural
    network to predict the mode content of the waveform given its parameters. Therefore,
    the results will vary compared to manually computing the mode selection, but it should be
    very accurate especially for the stronger modes.

    The mode filtering is a major contributing factor to the speed of these
    waveforms as it removes large numbers of useles modes from the final
    summation calculation.

    args:
        l_arr (1D int xp.ndarray): The l-mode indices for each mode index.
        m_arr (1D int xp.ndarray): The m-mode indices for each mode index.
        n_arr (1D int xp.ndarray): The n-mode indices for each mode index.
        threshold (double): The network threshold value for mode retention. Decrease to keep more modes,
            minimising missed modes but slowing down the waveform computation. Defaults to 0.5 (the optimal value for accuracy).

        **kwargs (dict, optional): Keyword arguments for the base classes:
            :class:`few.utils.baseclasses.ParallelModuleBase`.
            Default is {}.
    """

    def __init__(self, l_arr, m_arr, n_arr, threshold=0.5, **kwargs):
        ParallelModuleBase.__init__(self, **kwargs)

        # we set the pytorch device here for use with the neural network
        if self.use_gpu:
            self.xp = cp
            self.device=f"cuda:{cp.cuda.runtime.getDevice()}"
            self.neural_mode_list = [(lh, mh, nh) for lh, mh, nh in zip(l_arr.get(), m_arr.get(), n_arr.get())]
        else:
            self.xp = np
            self.device="cpu"
            self.neural_mode_list = [(lh, mh, nh) for lh, mh, nh in zip(l_arr, m_arr, n_arr)]

        few_dir = dir_path + "/../../"  # TODO proper file handling
        # TODO include waveform name in paths
        self.precomputed_mask = np.load(few_dir+"few/files/modeselector_files/FastSchwarzschildEccentricFluxBicubic/precomputed_mode_mask.npy")
        self.masked_mode_list = [self.neural_mode_list[maskind] for maskind in self.precomputed_mask]

        # import torch here in case users don't want it otherwise
        # if torch doesn't import properly, raise
        import torch
        self.torch = torch
        # if torch wasn't installed with GPU capability, raise
        if self.use_gpu and not torch.cuda.is_available():
            raise RuntimeError("pytorch has not been installed with CUDA capability. Fix installation or set use_gpu=False.")
        
        try:
            self.load_model(few_dir+"few/files/modeselector_files/FastSchwarzschildEccentricFluxBicubic/neural_mode_selector.tjm")
        except FileNotFoundError:
            raise FileNotFoundError("Neural mode predictor model file not found. ")

        self.threshold = threshold
        self.vector_min = np.load(few_dir+"few/files/modeselector_files/FastSchwarzschildEccentricFluxBicubic/vector_min.npy")
        self.vector_max = np.load(few_dir+"few/files/modeselector_files/FastSchwarzschildEccentricFluxBicubic/vector_max.npy")

    @property
    def gpu_capability(self):
        """Confirms GPU capability"""
        return True

    @property
    def is_predictive(self):
        """Whether this mode selector should be used before or after amplitude generation"""
        return True
    
    def attributes_ModeSelector(self):
        """
        attributes:
            xp: cupy or numpy depending on GPU usage.
            torch: pytorch module.
            device: The pytorch device currently in use: should agree with what is set by cupy.
            threshold: network evaluation threshold for keeping/discarding modes.
            vector_min: minimum bounds of the training data, for rescaling.
            vector_max: maximum bounds of the training data, for rescaling.
        """
        pass

    @property
    def citation(self):
        """Return citations related to this class."""
        return larger_few_citation + few_citation + few_software_citation

    def load_model(self, fp):
        self.model = self.torch.jit.load(fp)  # assume the model has been jit compiled
        self.model.to(self.device)
        self.model.eval()
    
    def __call__(self, M, mu, p0, e0, theta, phi, T, eps):
        """Call to predict the mode content of the waveform.

        This is the call function that takes the waveform parameters, applies a 
        precomputed mask and then evaluates the remaining mode content with a
        neural network classifier. 

        args:
            M (double): Mass of larger black hole in solar masses.
            mu (double): Mass of compact object in solar masses.
            p0 (double): Initial semilatus rectum (Must be greater than
                the separatrix at the the given e0 and x0).
                See documentation for more information on :math:`p_0<10`.
            e0 (double): Initial eccentricity.
            theta (double): Polar viewing angle.
            phi (double): Azimuthal viewing angle.
            T (double): Duration of waveform in years.
            eps (double): Mode selection threshold power.
        """

        #wrap angles to training bounds
        phi = phi % (2*np.pi)
        theta = theta % (np.pi) - np.pi

        inputs = np.array([[np.log(M), mu, p0, e0, T, theta, phi, np.log10(eps)]])
        # rescale network input from pre-computed
        inputs = 2 * (inputs - self.vector_min) / (self.vector_max - self.vector_min) - 1
        inputs = self.torch.as_tensor(inputs, device=self.device).float() 
        # get network output and threshold it based on the defined value
        with self.torch.inference_mode():
            mode_predictions = self.model(inputs)
            keep_inds = self.torch.where(mode_predictions > self.threshold)[0].int().cpu().numpy()
        # return list of modes for kept indices
        selected_modes = [self.masked_mode_list[ind] for ind in keep_inds]
        return selected_modes
