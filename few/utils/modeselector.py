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
from scipy import special

import os
from ..summation.interpolatedmodesum import CubicSplineInterpolant
from ..utils.citations import *
from ..utils.utility import get_fundamental_frequencies
from ..utils.constants import *
from ..utils.baseclasses import ParallelModuleBase
import sys

# check for cupy
try:
    import cupy as cp

except (ImportError, ModuleNotFoundError) as e:
    import numpy as np

# pytorch
try:
    import torch
except (ImportError, ModuleNotFoundError) as e:
    pass #  we can catch this later

from typing import Optional

dir_path = os.path.dirname(os.path.realpath(__file__))

@np.vectorize
def SPAFunc(x, th=7.0):
    II = 0.0 + 1.0j
    Gamp13 = 2.67893853470774763  # Gamma(1/3)
    Gamm13 = -4.06235381827920125  # Gamma(-1/3)

    if np.abs(x) <= th:
        xx = complex(x)
        pref1 = np.exp(-2. * np.pi * II / 3.) * pow(xx, 5. / 6.) * Gamm13 / pow(2., 1. / 3.)
        pref2 = np.exp(-np.pi * II / 3.) * pow(xx, 1. / 6.) * Gamp13 / pow(2., 2. / 3.)
        x2 = x * x

        c1_0, c1_2, c1_4, c1_6, c1_8, c1_10, c1_12, c1_14, c1_16, c1_18, c1_20, c1_22, c1_24, c1_26 = (
            0.5, -0.09375, 0.0050223214285714285714, -0.00012555803571428571429, 1.8109332074175824176e-6,
            -1.6977498819539835165e-8, 1.1169407118118312608e-10, -5.4396463237589184781e-13,
            2.0398673714095944293e-15, -6.0710338434809358015e-18, 1.4687985105195812423e-20,
            -2.9454515585285720100e-23, 4.9754249299469121790e-26, -7.1760936489618925658e-29
        )

        ser1 = c1_0 + x2*(c1_2 + x2*(c1_4 + x2*(c1_6 + x2*(c1_8 + x2*(c1_10 + x2*(c1_12 + x2*(c1_14 + x2*(c1_16 + x2*(c1_18 + x2*(c1_20 + x2*(c1_22 + x2*(c1_24 + x2*c1_26))))))))))))

        c2_0, c2_2, c2_4, c2_6, c2_8, c2_10, c2_12, c2_14, c2_16, c2_18, c2_20, c2_22, c2_24, c2_26 = (
            1., -0.375, 0.028125, -0.00087890625, 0.000014981356534090909091, -1.6051453429383116883e-7,
            1.1802539286311115355e-9, -6.3227889033809546546e-12, 2.5772237377911499951e-14,
            -8.2603324929203525483e-17, 2.1362928861000911763e-19, -4.5517604107246260858e-22,
            8.1281435905796894390e-25, -1.2340298973552160079e-27
        )

        ser2 = c2_0 + x2*(c2_2 + x2*(c2_4 + x2*(c2_6 + x2*(c2_8 + x2*(c2_10 + x2*(c2_12 + x2*(c2_14 + x2*(c2_16 + x2*(c2_18 + x2*(c2_20 + x2*(c2_22 + x2*(c2_24 + x2*c2_26))))))))))))

        ans = np.exp(-II * x) * (pref1 * ser1 + pref2 * ser2)
    else:
        y = 1. / x
        pref = np.exp(-0.75 * II * np.pi) * np.sqrt(0.5 * np.pi)

        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8 = (
            II, 0.069444444444444444444, -0.037133487654320987654 * II, -0.037993059127800640146,
            0.057649190412669721333 * II, 0.11609906402551541102, -0.29159139923075051147 * II,
            -0.87766696951001691647, 3.0794530301731669934 * II
        )

        ser = c_0 + y * (c_1 + y * (c_2 + y * (c_3 + y * (c_4 + y * (c_5 + y * (c_6 + y * (c_7 + y * c_8)))))))

        ans = pref * ser

    return ans

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

    def __init__(self, l_arr: np.ndarray, m_arr: np.ndarray, n_arr: np.ndarray, sensitivity_fn: Optional[object]=None, **kwargs: Optional[dict]):
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

    def __call__(self, teuk_modes: np.ndarray, ylms: np.ndarray, modeinds: list[np.ndarray], fund_freq_args: Optional[tuple], eps: float=1e-5) -> np.ndarray:
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

        zero_modes_mask = (modeinds[1]==0)*(modeinds[2]==0)
        
        # get the power contribution of each mode including m < 0
        if self.sensitivity_fn is None:
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
        elif self.sensitivity_fn is not None:
            
            if fund_freq_args is None:
                raise ValueError(
                    "If sensitivity weighting is desired, the fund_freq_args kwarg must be provided."
                )

            M = fund_freq_args[0]
            Msec = M * MTSUN_SI

            a_fr, p_fr, e_fr, x_fr = fund_freq_args[1:-1]

            if self.use_gpu:  # fundamental frequencies only defined on CPU
                p_fr = p_fr.get()
                e_fr = e_fr.get()
                x_fr = x_fr.get()
            # get dimensionless fundamental frequency
            OmegaPhi, OmegaTheta, OmegaR = get_fundamental_frequencies(
                a_fr, p_fr, e_fr, x_fr
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

            # weight by PSD, only for non zero modes    
            cs = CubicSplineInterpolant(fund_freq_args[-1], freqs[:,~zero_modes_mask].T, use_gpu=self.use_gpu)
            fdot = cs(fund_freq_args[-1],deriv_order=1)
            fddot = cs(fund_freq_args[-1],deriv_order=2)
            arg_1 = 2*np.pi*fdot**3 / (3*fddot**2)
            try:
                arg_1_cpu = arg_1.get()
            except:
                arg_1_cpu = arg_1
            spa_func = xp.asarray(SPAFunc(arg_1_cpu))
            fact = -1*fdot/xp.abs(fddot) * spa_func * 2./np.sqrt(3) / xp.sqrt(arg_1+0j)
 

            ### after square
            power = (
                xp.abs(
                    xp.concatenate(
                        [teuk_modes, xp.conj(teuk_modes[:, self.m0mask])], axis=1
                    )
                    * ylms)
                **2
            )
            power[:,~zero_modes_mask] /= PSD[:,~zero_modes_mask] / xp.abs(fact).T

            ### before square
            # amplitudes = (
            #     xp.concatenate(
            #         [teuk_modes, xp.conj(teuk_modes[:, self.m0mask])], axis=1
            #     )
            #     * ylms
            # )
            # power = xp.abs(amplitudes[:,~zero_modes_mask] / fact.T)**2
            # power /= PSD[:,~zero_modes_mask]

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

    def __init__(self, l_arr, m_arr, n_arr, threshold=0.5, mode_selector_location=None, return_type="tuples", keep_inds=None, maximise_over_theta=False, **kwargs):
        if mode_selector_location is None:
            raise ValueError("mode_selector_location kwarg cannot be none.")
        elif not os.path.isdir(mode_selector_location):
            raise ValueError(f"mode_selector location path ({mode_selector_location}) does not point to an existing directory.")        

        ParallelModuleBase.__init__(self, **kwargs)

        # we set the pytorch device here for use with the neural network
        if self.use_gpu:
            xp = cp
            self.device=f"cuda:{cp.cuda.runtime.getDevice()}"
            self.neural_mode_list = [(lh, mh, nh) for lh, mh, nh in zip(l_arr.get(), m_arr.get(), n_arr.get())]
        else:
            xp = np
            self.device="cpu"
            self.neural_mode_list = [(lh, mh, nh) for lh, mh, nh in zip(l_arr, m_arr, n_arr)]
        
        # TODO include waveform name in paths
        self.precomputed_mask = np.load(os.path.join(mode_selector_location, "precomputed_mode_mask.npy")).astype(np.int32)
        self.masked_mode_list = [self.neural_mode_list[maskind] for maskind in self.precomputed_mask]

        self.precomputed_mask = xp.asarray(self.precomputed_mask)  # for "array" return type compatibility

        if "torch" not in sys.modules:
            raise ModuleNotFoundError("Pytorch not installed.")
        # if torch wasn't installed with GPU capability, raise
        if self.use_gpu and not torch.cuda.is_available():
            raise RuntimeError("pytorch has not been installed with CUDA capability. Fix installation or set use_gpu=False.")
        
        self.model_loc = os.path.join(mode_selector_location, "neural_mode_selector.tjm")
        try:
            self.load_model(self.model_loc)
        except FileNotFoundError:
            raise FileNotFoundError("Neural mode predictor model file not found.")

        self.threshold = threshold
        self.vector_min, self.vector_max = np.load(os.path.join(mode_selector_location, "network_norm.npy"))

        self.return_type = return_type

        self.keep_inds = keep_inds

        self.maximise_over_theta = maximise_over_theta

    @property
    def gpu_capability(self):
        """Confirms GPU capability"""
        return True

    @property
    def is_predictive(self):
        """Whether this mode selector should be used before or after amplitude generation"""
        return True
    
    def __getstate__(self):
        # needs to be modified for pickleability
        if hasattr(self, "model"):
            del self.model
        return self.__dict__
    
    def __setstate__(self, d):
        self.__dict__ = d
        # we now load the model as we've unpickled again
        self.load_model(self.model_loc)

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
        self.model = torch.jit.load(fp)  # assume the model has been jit compiled
        self.model.to(self.device)
        self.model.eval()
    
    def __call__(self, M, mu, a, p0, e0, xI, theta, phi, T, eps):
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
        
        if not hasattr(self, "model"):
            self.load_model(self.model_loc)

        if self.use_gpu:
            xp = cp
        else:
            xp = np

        #wrap angles to training bounds
        phi = phi % (2*np.pi)

        # maximise mode count over theta (stabilises the likelihood but often requires more modes)
        if self.maximise_over_theta:
            theta = 1.39

        inputs = np.asarray([[np.log(M), mu, a, p0, e0, xI, T, theta, phi, np.log10(eps)]])[:,np.asarray(self.keep_inds)]  # throw away the params we dont need
        # rescale network input from pre-computed
        inputs = 2 * (inputs - self.vector_min) / (self.vector_max - self.vector_min) - 1
        inputs = torch.as_tensor(inputs, device=self.device).float() 
        self.check_bounds(inputs)
        # get network output and threshold it based on the defined value
        with torch.inference_mode():
            mode_predictions = torch.nn.functional.sigmoid(self.model(inputs))
            keep_inds = torch.where(mode_predictions > self.threshold)[0].int().cpu().numpy()  # cpu() works for cpu and gpu
        # return list of modes for kept indices
        if self.return_type == "tuples":
            return [self.masked_mode_list[ind] for ind in keep_inds]
        elif self.return_type == "array":
            return self.precomputed_mask[keep_inds]

    def check_bounds(self, inputs):
        # Enforce bounds. Normalising to [-1,1] so it's easy to check (space for numerical errors)
        if torch.any(torch.abs(inputs) > 1 + 1e-3):
            breakpoint()
            raise ValueError(f"One of the inputs to the neural mode selector is out of bounds. Normalised inputs: {inputs}")