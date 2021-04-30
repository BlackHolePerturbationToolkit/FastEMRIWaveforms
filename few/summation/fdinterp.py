# Interpolated summation of modes in python for the FastEMRIWaveforms Package

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

import warnings

import numpy as np

# try to import cupy
try:
    import cupy as xp

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp

# Cython imports
from pyinterp_cpu import interpolate_arrays_wrap as interpolate_arrays_wrap_cpu
from pyinterp_cpu import get_waveform_wrap as get_waveform_wrap_cpu

# Python imports
from few.utils.baseclasses import SummationBase, SchwarzschildEccentric, GPUModuleBase
from few.utils.citations import *
from few.utils.utility import get_fundamental_frequencies
from few.utils.constants import *
from few.summation.interpolatedmodesum import CubicSplineInterpolant

# Attempt Cython imports of GPU functions
try:
    from pyinterp import interpolate_arrays_wrap, get_waveform_wrap

except (ImportError, ModuleNotFoundError) as e:
    pass

# for special functions
from scipy import special
from scipy.interpolate import CubicSpline


# added to return element or none in list
def find_element_in_list(element, list_element):
    try:
        index_element = list_element.index(element)
        return index_element
    except ValueError:
        return None



class FDInterpolatedModeSum(SummationBase, SchwarzschildEccentric, GPUModuleBase):
    """Create waveform by interpolating sparse trajectory in the frequency domain.

    It interpolates all of the modes of interest and phases at sparse
    trajectories. Within the summation phase, the values are calculated using
    the interpolants and summed.

    This class can be run on GPUs and CPUs.
    """
    
    def __init__(self, *args, **kwargs):

        GPUModuleBase.__init__(self, *args, **kwargs)
        SchwarzschildEccentric.__init__(self, *args, **kwargs)
        SummationBase.__init__(self, *args, **kwargs)

        self.kwargs = kwargs

        # eventually the name will change to adapt for the gpu implementantion
        if self.use_gpu:
            self.get_waveform = get_waveform_wrap

        else:
            self.get_waveform = get_waveform_wrap_cpu

    def attributes_FDInterpolatedModeSum(self):
        """
        attributes:
            get_waveform (func): CPU or GPU function for waveform creation.

        """

    def waveform_spa_factors(self, fdot_spline, fddot_spline):

        # Waveform Amplitudes
        arg = -1j* 2*np.pi*fdot_spline**3 / (3*fddot_spline**2)
        K_1over3 = special.kve(1./3.,arg) #special.kv(1./3.,arg)*np.exp(arg)
        
        # to correct the special function nan
        if np.sum(np.isnan(special.kv(1/3,arg))):
            X = 2*np.pi*fdot_spline**3 / (3*fddot_spline**2)
            K_1over3[np.isnan(special.kv(1/3,arg))] = 0.0 # (np.sqrt(np.pi/2) /(1j*np.sqrt(np.abs(X))) * np.exp(-1j*np.pi/4) )[np.isnan(special.kv(1/3,arg))]

        return 1j* fdot_spline/np.abs(fddot_spline) * K_1over3 * 2/np.sqrt(3)


    def time_frequency_map(self,spline_f_mode,index_star):
        # turn over index
        index_star = index_star - 1 

        t = spline_f_mode.t
        f_mn = spline_f_mode.y[-1]

        # find t_star analytically
        ratio = spline_f_mode.c2[0,index_star]/(3*spline_f_mode.c3[0,index_star])
        second_ratio = spline_f_mode.c1[0,index_star]/(3*spline_f_mode.c3[0,index_star])
        t_star = t[index_star] -ratio + np.sqrt(ratio**2 - second_ratio)

        # new frequancy vector
        Fstar = spline_f_mode(t_star)
        new_F = np.append(f_mn[:index_star+1], Fstar + Fstar/np.abs(Fstar) * np.abs(f_mn[index_star+1:] - Fstar) )

        # new t_f
        sign_slope = spline_f_mode.c1[0,0]/np.abs(spline_f_mode.c1[0,0])

        # frequency split
        initial_frequency = f_mn[0]
        end_frequency = f_mn[-1]


        if (initial_frequency>end_frequency):

            if sign_slope>0:
                # alt imple
                modified2_t_f = CubicSpline((new_F),t)
                ind_1 = (self.frequency>end_frequency)*(self.frequency<Fstar)
                t_f_1 = modified2_t_f(Fstar+Fstar/np.abs(Fstar) * np.abs(self.frequency[ind_1] - Fstar))
                ind_2 = (self.frequency>initial_frequency)*(self.frequency<Fstar)
                t_f_2 = modified2_t_f(self.frequency[ind_2])

            else:
                modified2_t_f = CubicSpline(-(new_F),t) # 
                ind_1 = (self.frequency<end_frequency)*(self.frequency>Fstar)
                ind_2 = (self.frequency<initial_frequency)*(self.frequency>Fstar)
                t_f_1 = modified2_t_f(-(Fstar+Fstar/np.abs(Fstar) * np.abs(self.frequency[ind_1] - Fstar)))
                t_f_2 = modified2_t_f(-(self.frequency[ind_2]))


        if (initial_frequency<end_frequency):
            
            if sign_slope>0:

                modified2_t_f = CubicSpline((new_F),t)

                ind_1 = (self.frequency>initial_frequency)*(self.frequency<Fstar)
                ind_2 = (self.frequency>end_frequency)*(self.frequency<Fstar)
                t_f_2 = modified2_t_f((Fstar+Fstar/np.abs(Fstar) * np.abs(self.frequency[ind_2] - Fstar)))
                t_f_1 = modified2_t_f((self.frequency[ind_1]))

            else:
                modified2_t_f = CubicSpline(-(new_F),t) # 
                
                ind_1 = (self.frequency<end_frequency)*(self.frequency>Fstar)
                ind_2 = (self.frequency<initial_frequency)*(self.frequency>Fstar)
                t_f_1 = modified2_t_f(-(Fstar+Fstar/np.abs(Fstar) * np.abs(self.frequency[ind_1] - Fstar)))
                t_f_2 = modified2_t_f(-(self.frequency[ind_2]))


        return ind_1, ind_2, t_f_1, t_f_2

    @property
    def gpu_capability(self):
        """Confirms GPU capability"""
        return True

    @property
    def citation(self):
        return larger_few_citation + few_citation + few_software_citation

    def sum(
        self,
        t,
        teuk_modes,
        ylms,
        Phi_phi,
        Phi_r,
        m_arr,
        n_arr,
        M,
        p,
        e,
        *args,
        dt=10.0,
        **kwargs,
    ):
        """Interpolated summation function.

        This function interpolates the amplitude and phase information, and
        creates the final waveform with the combination of ylm values for each
        mode.

        args:
            t (1D double xp.ndarray): Array of t values.
            teuk_modes (2D double xp.array): Array of complex amplitudes.
                Shape: (len(t), num_teuk_modes).
            ylms (1D complex128 xp.ndarray): Array of ylm values for each mode,
                including m<0. Shape is (num of m==0,) + (num of m>0,)
                + (num of m<0). Number of m<0 and m>0 is the same, but they are
                ordered as (m==0 first then) m>0 then m<0.
            Phi_phi (1D double xp.ndarray): Array of azimuthal phase values
                (:math:`\Phi_\phi`).
            Phi_r (1D double xp.ndarray): Array of radial phase values
                 (:math:`\Phi_r`).
            m_arr (1D int xp.ndarray): :math:`m` values associated with each mode.
            n_arr (1D int xp.ndarray): :math:`n` values associated with each mode.
            *args (list, placeholder): Added for future flexibility.
            dt (double, optional): Time spacing between observations (inverse of sampling
                rate). Default is 10.0.
            **kwargs (dict, placeholder): Added for future flexibility.

        """

        init_len = len(t) # length of sparse traj
        num_teuk_modes = teuk_modes.shape[1] 
        num_pts = self.num_pts # from the base clase, adjust in the baseclass

        length = init_len
        # number of quantities to be interp in time domain
        ninterps = (2 * self.ndim) + 2 * num_teuk_modes  # 2 for re and im
        y_all = self.xp.zeros((ninterps, length)) 

        y_all[:num_teuk_modes] = teuk_modes.T.real
        y_all[num_teuk_modes : 2 * num_teuk_modes] = teuk_modes.T.imag

        y_all[-2] = Phi_phi
        y_all[-1] = Phi_r

        try:
            p, e = p.get(), e.get()
        except AttributeError:
            pass

        Omega_phi, Omega_theta, Omega_r = get_fundamental_frequencies(
            0.0, p, e, np.zeros_like(e)
        )

        f_phi, f_r = (
            Omega_phi / (2 * np.pi * M * MTSUN_SI),
            Omega_r / (2 * np.pi * M * MTSUN_SI),
        )

        y_all[-4] = f_phi
        y_all[-3] = f_r

        spline = CubicSplineInterpolant(t, y_all, use_gpu=self.use_gpu)

        # this should be the same
        # plt.plot(t,Omega_phi/(M * MTSUN_SI)); plt.plot(t,spline.deriv(t)[-2],'--' ); plt.show()

        # define a frequency vector
        self.frequency = np.fft.fftfreq(self.num_pts + self.num_pts_pad,dt)

        # final waveform
        h = np.zeros_like(self.frequency, dtype=complex)

        # indentify where there is a turnover
        for j, (m, n) in enumerate(zip(m_arr, n_arr)):

            # check turnover
            freq_mode_start = m * spline.c1[-4,0] + n * spline.c1[-3,0]
            index_star = find_element_in_list(True, list(freq_mode_start*(m * spline.c1[-4,:] + n * spline.c1[-3,:])<0.))
            
            # frequency mode
            f_mn = m * f_phi + n * f_r
            spline_f_mode = CubicSplineInterpolant(t, f_mn, use_gpu=self.use_gpu)

            min_f_mn = np.min(np.abs(f_mn))
            max_f_mn = np.max(np.abs(f_mn))

            # fdot
            fdot_spline = CubicSplineInterpolant(t, np.asarray(spline_f_mode.deriv(t)), use_gpu=self.use_gpu)

            # fddot
            fddot_spline = CubicSplineInterpolant(t, np.asarray(fdot_spline.deriv(t)), use_gpu=self.use_gpu)

            if index_star is not None:
                print(m,n)
                print('there is a turn-over')

                # calculate frequency indeces and respective time of f
                ind_1, ind_2, t_f_1, t_f_2 = self.time_frequency_map(spline_f_mode,index_star)
                
                """
                # turn over index
                index_star = index_star - 1 


                # find t_star analytically
                ratio = spline_f_mode.c2[0,index_star]/(3*spline_f_mode.c3[0,index_star])
                second_ratio = spline_f_mode.c1[0,index_star]/(3*spline_f_mode.c3[0,index_star])
                t_star = t[index_star] -ratio + np.sqrt(ratio**2 - second_ratio)

                # new frequancy vector
                Fstar = spline_f_mode(t_star)
                new_F = np.append(f_mn[:index_star+1], Fstar + Fstar/np.abs(Fstar) * np.abs(f_mn[index_star+1:] - Fstar) )
                new_F_spline = CubicSplineInterpolant(t, new_F)

                # new t_f
                sign_slope = spline_f_mode.c1[0,0]/np.abs(spline_f_mode.c1[0,0])
                # HERE PROBLEM WITH THE CUBIC SPLINEInterp
                # should throw a warning if decreasing
                # need to put an absolute value but it might be incorrect for modes passing through zero
                modified_t_f = CubicSpline(np.abs(new_F),t) # take axis frequency monotonically increasing

                # frequency split
                initial_frequency = f_mn[0]
                end_frequency = f_mn[-1]


                if (initial_frequency>end_frequency):
                    index_single = (self.frequency<initial_frequency)*(self.frequency>end_frequency)

                    if sign_slope>0:
                        # alt imple
                        modified2_t_f = CubicSpline((new_F),t)
                        ind_1 = (self.frequency>end_frequency)*(self.frequency<Fstar)
                        t_f_1 = modified2_t_f(Fstar+Fstar/np.abs(Fstar) * np.abs(self.frequency[ind_1] - Fstar))
                        ind_2 = (self.frequency>initial_frequency)*(self.frequency<Fstar)
                        t_f_2 = modified2_t_f(self.frequency[ind_2])

                        index_double = (self.frequency>initial_frequency)*(self.frequency<Fstar)
                        t_f_sing = modified_t_f(np.abs(Fstar+Fstar/np.abs(Fstar) * np.abs(self.frequency[index_single] - Fstar)))
                    else:
                        modified2_t_f = CubicSpline(-(new_F),t) # 
                        
                        ind_1 = (self.frequency<end_frequency)*(self.frequency>Fstar)
                        ind_2 = (self.frequency<initial_frequency)*(self.frequency>Fstar)
                        t_f_1 = modified2_t_f(-(Fstar+Fstar/np.abs(Fstar) * np.abs(self.frequency[ind_1] - Fstar)))
                        t_f_2 = modified2_t_f(-(self.frequency[ind_2]))

                        index_double = (self.frequency<end_frequency)*(self.frequency>Fstar)
                        t_f_sing = modified_t_f(np.abs(self.frequency[index_single]))


                if (initial_frequency<end_frequency):
                    index_single = (self.frequency>initial_frequency)*(self.frequency<end_frequency)
                    
                    if sign_slope>0:

                        modified2_t_f = CubicSpline((new_F),t)

                        ind_1 = (self.frequency>initial_frequency)*(self.frequency<Fstar)
                        ind_2 = (self.frequency>end_frequency)*(self.frequency<Fstar)
                        t_f_2 = modified2_t_f((Fstar+Fstar/np.abs(Fstar) * np.abs(self.frequency[ind_2] - Fstar)))
                        t_f_1 = modified2_t_f((self.frequency[ind_1]))

                        #index_double = (self.frequency>end_frequency)*(self.frequency<Fstar)
                        #t_f_sing = modified_t_f(np.abs(self.frequency[index_single]))
                    else:
                        modified2_t_f = CubicSpline(-(new_F),t) # 
                        
                        ind_1 = (self.frequency<end_frequency)*(self.frequency>Fstar)
                        ind_2 = (self.frequency<initial_frequency)*(self.frequency>Fstar)
                        t_f_1 = modified2_t_f(-(Fstar+Fstar/np.abs(Fstar) * np.abs(self.frequency[ind_1] - Fstar)))
                        t_f_2 = modified2_t_f(-(self.frequency[ind_2]))

                        index_double = (self.frequency<initial_frequency)*(self.frequency>Fstar)
                        t_f_sing = modified_t_f(np.abs(Fstar+Fstar/np.abs(Fstar) * np.abs(self.frequency[index_single] - Fstar)))

                
                # plt.plot(t,f_mn); plt.axhline(self.frequency[index_single][0]); plt.show()
                # plt.plot(t,f_mn); plt.axhline(self.frequency[index_single][-1]); plt.show()
                # import matplotlib.pyplot as plt
                # print(spline(t_f_sing))

                
                # double contribution
                t_f_first_half = modified_t_f( np.abs(self.frequency[index_double])  )
                t_f_second_half = modified_t_f(np.abs(Fstar+Fstar/np.abs(Fstar) * np.abs(self.frequency[index_double] - Fstar)))
                
                # Waveform Amplitudes

                h_two_contr = [(spline(t_two)[j] + 1j* spline(t_two)[num_teuk_modes+j] )*ylms[j]*\
                                    self.waveform_spa_factors(fdot_spline(t_two), fddot_spline(t_two))*\
                                    np.exp( 1j*( 2*np.pi*f* t_two  - ( m * spline(t_two)[-2] + n * spline(t_two)[-1]) ) )
                                for t_two,f in zip([t_f_first_half,t_f_second_half,t_f_sing],[self.frequency[index_double],self.frequency[index_double],self.frequency[index_single]])
                                ]
                """

                #print(sign_slope,(initial_frequency<end_frequency) )
                #breakpoint()
                h_contr = [(spline(t_two)[j] + 1j* spline(t_two)[num_teuk_modes+j] )*ylms[j]*\
                    self.waveform_spa_factors(fdot_spline(t_two), fddot_spline(t_two))*\
                    np.exp( 1j*( 2*np.pi*f* t_two  - ( m * spline(t_two)[-2] + n * spline(t_two)[-1]) ) )
                for t_two,f in zip([t_f_1,t_f_2],[self.frequency[ind_1],self.frequency[ind_2]])
                ]

                h[ind_1] += h_contr[0]
                h[ind_2] += h_contr[1]
                #h[index_double] += np.sum(h_two_contr[:2],axis=0)
                #h[index_single] += h_two_contr[2]


            else:

                # time of f
                t_of_f = CubicSplineInterpolant(f_mn, t, use_gpu=self.use_gpu)
                
                # frequency
                index_positive_f = (self.frequency>min_f_mn)*(self.frequency<max_f_mn)
                index_negative_f = (self.frequency<-min_f_mn)*(self.frequency>-max_f_mn)
                f_pos = self.frequency[index_positive_f]
                f_neg = self.frequency[index_negative_f]

                # time evluated quantities
                t_pos = t_of_f(f_pos)
                t_neg = t_of_f(-f_neg) # same as np.flip(t_pos)


                # teukolsky modes
                Teuk_sph_pos = (spline(t_pos)[j] + 1j* spline(t_pos)[num_teuk_modes+j] )*ylms[j]*\
                    self.waveform_spa_factors(fdot_spline(t_pos), fddot_spline(t_pos))

                Teuk_sph_neg = (spline(t_neg)[j] - 1j* spline(t_neg)[num_teuk_modes+j] )*ylms[j+num_teuk_modes]*\
                    self.waveform_spa_factors(-fdot_spline(t_neg), -fddot_spline(t_neg))

                
                Exp_pos = np.exp(1j*(2*np.pi* f_pos * t_pos  - (m * spline(t_pos)[-2] + n * spline(t_pos)[-1]) ) )
                
                Exp_neg = np.exp(1j*(2*np.pi* f_neg * t_neg  + (m * spline(t_neg)[-2] + n * spline(t_neg)[-1]) ) )

                h[index_positive_f] += Teuk_sph_pos*Exp_pos
                h[index_negative_f] += Teuk_sph_neg*Exp_neg


        self.waveform = h
