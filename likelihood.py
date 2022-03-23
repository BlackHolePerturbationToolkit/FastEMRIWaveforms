dev = 7
import os
os.system(f"CUDA_VISIBLE_DEVICES={dev}")
os.environ["CUDA_VISIBLE_DEVICES"] = f"{dev}"
os.system("echo $CUDA_VISIBLE_DEVICES")

os.system("export OMP_NUM_THREADS=1")
os.environ["OMP_NUM_THREADS"] = "1"

import unittest
import numpy as np
import warnings

from few.trajectory.inspiral import EMRIInspiral
from few.amplitude.romannet import RomanAmplitude
from few.amplitude.interp2dcubicspline import Interp2DAmplitude
from few.waveform import FastSchwarzschildEccentricFlux, SlowSchwarzschildEccentricFlux, FastSchwarzschildEccentricFluxHarmonics
from few.utils.utility import get_overlap, get_mismatch
from few.utils.ylm import GetYlms
from few.utils.modeselector import ModeSelector
from few.summation.interpolatedmodesum import CubicSplineInterpolant
from few.utils.utility import get_fundamental_frequencies
from few.utils.constants import *
from few.waveform import GenerateEMRIWaveform

from scipy.interpolate import CubicSpline

try:
    import cupy as xp

    gpu_available = True

except (ModuleNotFoundError, ImportError) as e:
    import numpy as xp

    warnings.warn(
        "CuPy is not installed or a gpu is not available. If trying to run on a gpu, please install CuPy."
    )
    gpu_available = False

# process FD

inspiral_kwargs = {
    "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
    "max_init_len": int(
        1e3
    ),  # all of the trajectories will be well under len = 1000
}

# keyword arguments for inspiral generator (RomanAmplitude)
amplitude_kwargs = {
    "max_init_len": int(
        1e3
    )  # all of the trajectories will be well under len = 1000
}

# keyword arguments for Ylm generator (GetYlms)
Ylm_kwargs = {
    "assume_positive_m": False  # if we assume positive m, it will generate negative m for all m>0
}

# keyword arguments for summation generator (InterpolatedModeSum)
sum_kwargs = dict(pad_output=True, output_type="fd")

# list of waveforms
fast = FastSchwarzschildEccentricFluxHarmonics(
    inspiral_kwargs=inspiral_kwargs,
    amplitude_kwargs=amplitude_kwargs,
    Ylm_kwargs=Ylm_kwargs,
    sum_kwargs=sum_kwargs,
    use_gpu=gpu_available,
)

class fd_waveform():

    def __init__(self, wavemodule, use_gpu=False, N=int(1e7+1)):
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.xp = xp
        else:
            self.xp = np

        self.wave = wavemodule
        self.f_arr = self.xp.linspace(-1 / (2 * dt), +1 / (2 * dt), num= N)
        self.f_in = self.f_arr[int(( N - 1 ) / 2 + 1):]
        self.psd = self.PowerSpectralDensity(self.f_in)/(self.f_in[1]-self.f_in[0])
        self.psd[0]=self.psd[1]

    def __call__(self, *args, **kwargs):
        
        kwargs["f_arr"] = self.f_arr

        return self.transform_to_fft_hp_hcross(self.wave(*args, **kwargs))
    
    def inject_signal(self, inds, *inj_params, **kwargs):
        self.inds = inds
        self.inj_params = inj_params
        self.wave_kwargs = kwargs

        self.d = self.__call__(*inj_params, **self.wave_kwargs)
        self.d_d = self.InnerProduct(self.d, self.d)#sum([self.InnerProduct(data, data) for data in self.d])

        self.set_up_RelativeBinning()

        return self.d

    def get_ll(self, params):

        full_params = np.asarray(self.inj_params)
        full_params[self.inds] = params

        h = self.__call__(*full_params, **self.wave_kwargs)

        h_h = self.InnerProduct(h, h)
        d_h = self.InnerProduct(h,self.d)

        like_out = -1.0 / 2.0 * (self.d_d + h_h - 2.0 * d_h).real
        print("d_h", d_h, "h_h", h_h, "like", like_out)

        # back to CPU if on GPU
        try:
            return like_out.get()

        except AttributeError:
            return like_out

    def set_up_RelativeBinning(self):

        # pre-computation
        # reference waveform
        ref_params = np.asarray(self.inj_params)

        # TODO keep only monotonic harmonics
        self.mod_sel = [[(int(l),int(m),int(n))] for l,m,n in zip(self.wave.waveform_generator.ls, self.wave.waveform_generator.ms, self.wave.waveform_generator.ns)]

        # get inspiral generator from waveform
        self.new_t=np.linspace(0.0, 365*3600*24*self.wave_kwargs["T"] ,num=100)
        self.inspiral = self.wave.waveform_generator.inspiral_generator
        t, p, e, x, Phi_phi, Phi_theta, Phi_r = self.inspiral(*ref_params[:6], T=self.wave_kwargs["T"])#, new_t=self.new_t)
        Omega_phi, Omega_theta, Omega_r = get_fundamental_frequencies(
                    0.0, p, e, np.zeros_like(e)
                )

        f_phi, f_r = (
                    self.xp.asarray(Omega_phi / (2 * np.pi * M * MTSUN_SI)),
                    self.xp.asarray(Omega_r / (2 * np.pi * M * MTSUN_SI)),
                )
        # frequency range per each harmonic
        f_range = self.xp.array([m*f_phi + n*f_r for m,n in zip(self.wave.waveform_generator.ms, self.wave.waveform_generator.ns)])
        phase_evolution = self.xp.array([m* self.xp.asarray(Phi_phi) + n* self.xp.asarray(Phi_r) for m,n in zip(self.wave.waveform_generator.ms, self.wave.waveform_generator.ns)])

        # delete eps if present
        self.new_kw = self.wave_kwargs.copy()
        try:
            del self.new_kw["eps"]
        except:
            pass
        try:
            del self.new_kw["mode_selection"]
        except:
            pass
        
        # list of harmonics
        self.ref_wave_mode_pol = self.xp.asarray([self.__call__(*ref_params, **self.new_kw, mode_selection=md) for md in self.mod_sel])

        # A vector shape (len(self.mod_sel) , 2,  3)
        self.A_vec = self.xp.array([[self.A_vector(self.d[0], wave_mode[0]), self.A_vector(self.d[1], wave_mode[1])] for wave_mode in self.ref_wave_mode_pol])
        # B vector shape (2, (len(self.mod_sel)+1)*len(self.mod_sel)/2 , 5)
        self.B_matrix_p = self.xp.array([[self.B_vector(self.ref_wave_mode_pol[m1,pol,:], self.ref_wave_mode_pol[m2,pol,:]) for m1 in range(len(self.mod_sel)) for m2 in range(len(self.mod_sel)) if m1<=m2] for pol in range(2) ])
        
        # frequency of each bin
        freq = self.f_in
        f_bin = [freq[((xp.min(fr)<freq)*(freq<xp.max(fr)))] for fr in f_range]
        # take some points in the frequency bin
        self.f_to_eval = [self.xp.array([fb[int(len(fb)*0.2)], fb[int(len(fb)*0.5)], fb[int(len(fb)*0.8)]]) for fb in f_bin]
        self.ind_f = [self.xp.array([xp.where(freq==ff[i])[0] for i in range(3)]).flatten() for ff in self.f_to_eval]

        # approximate ratio
        self.f_sym = self.xp.asarray([self.xp.sort(self.xp.hstack((ftemp,0,-ftemp) )) for ftemp in self.f_to_eval])
        self.ref_wave_noAmp = self.xp.asarray([self.phase_ev( t, f_range[i], phase_evolution[i], self.f_sym[i]) for i in range(len(f_range))])

    def get_approximate_ratios(self, *par):
        # get inspiral generator from waveform
        t, p, e, x, Phi_phi, Phi_theta, Phi_r = self.inspiral(*par, T=self.wave_kwargs["T"])#, new_t=self.new_t)
        Omega_phi, Omega_theta, Omega_r = get_fundamental_frequencies(
                    0.0, p, e, np.zeros_like(e)
                )

        f_phi, f_r = (
                    self.xp.asarray(Omega_phi / (2 * np.pi * M * MTSUN_SI)),
                    self.xp.asarray(Omega_r / (2 * np.pi * M * MTSUN_SI)),
                )
        # frequency range per each harmonic
        m_n_arr = self.xp.asarray(self.mod_sel)
        m_arr = m_n_arr[:,:,1].flatten()
        n_arr = m_n_arr[:,:,2].flatten()
        f_range = self.xp.array([m*f_phi + n*f_r for m,n in zip(m_arr, n_arr)])
        phase_evolution = self.xp.array([m * self.xp.asarray(Phi_phi) + n * self.xp.asarray(Phi_r) for m,n in zip(m_arr, n_arr)])
        self.h_noAmp = self.xp.asarray([self.phase_ev( t, f_range[i], phase_evolution[i], self.f_sym[i]) for i in range(len(f_range))])
        return self.h_noAmp/self.ref_wave_noAmp

    def phase_ev(self, t, frequency_evolution, phase_evolution, f_array):

        phase_spline = CubicSplineInterpolant(t, phase_evolution, use_gpu=self.use_gpu)

        time_f_spline_0 = CubicSplineInterpolant(frequency_evolution, t, use_gpu=self.use_gpu)

        # frequency
        index_positive_f = (f_array>self.xp.min(frequency_evolution))*(f_array<self.xp.max(frequency_evolution))
        index_negative_f = self.xp.flip(index_positive_f)
        # index_negative_f = (freq>self.xp.min(-frequency_evolution))*(freq<self.xp.max(-frequency_evolution))
        f_0 = f_array[index_positive_f]
        f_1 = f_array[index_negative_f]

        # time evluated quantities
        t_f_0 = time_f_spline_0(f_0)
        t_f_1 = self.xp.flip(t_f_0)#time_f_spline_0(-f_1)#

        # fdot
        fdot_spline_0 = phase_spline(t_f_0, deriv_order=2)
        fdot_spline_1 = -phase_spline(t_f_1, deriv_order=2)

        Exp0 = self.xp.exp(-1j*(2*self.xp.pi*f_0* t_f_0  - phase_spline(t_f_0)- self.xp.pi/4) ) / self.xp.sqrt(self.xp.abs(fdot_spline_0))
        Exp1 = self.xp.exp(-1j*(2*self.xp.pi*f_1* t_f_1  + phase_spline(t_f_1)+ self.xp.pi/4) ) / self.xp.sqrt(self.xp.abs(fdot_spline_1))

        # final waveform
        h = self.xp.zeros_like(f_array, dtype=complex) 
        h[index_positive_f] = Exp0
        h[index_negative_f] = Exp1

        fd_sig = self.xp.flip(-h )
        fft_sig_r = self.xp.real(fd_sig + self.xp.flip(fd_sig) )/2.0 + 1j * self.xp.imag(fd_sig - self.xp.flip(fd_sig))/2.0
        ind = int(( len(f_array) - 1 ) / 2 + 1)
        final_h = self.xp.real(fft_sig_r[ind:]) - 1j* self.xp.imag(fft_sig_r[ind:])

        return final_h/ Gpc * MRSUN_SI

    def get_RB_ll(self, params, approximate_ratio=False):
        
        # -------------- ONLINE computation -----------------------
        full_params = np.asarray(self.inj_params)
        full_params[self.inds] = params
        
        if approximate_ratio:
            app_ratios = self.get_approximate_ratios(*full_params[:6])
            bb = self.xp.empty((len(self.mod_sel),3,2), dtype=complex) 
            bb[:,:,0] = app_ratios
            bb[:,:,1] = app_ratios
        else:
            # get template, notice I am generating the inspiral multiple times, this should be avoided!
            h_wave_mode_pol = self.xp.asarray([self.__call__(*full_params, **self.new_kw, mode_selection=md) for md in self.mod_sel])
            # get left hand side of M r = b
            bb = self.xp.array([h_wave_mode_pol[i,:,self.ind_f[i]]/self.ref_wave_mode_pol[i,:,self.ind_f[i]] for i in range(len(self.mod_sel))])
            # np.abs ( (bb[:,:,0]-bb[:,:,1])/bb[:,:,1] ) h plus and cross have equal ratios
        
        # check against the true ones
        # for i in range(len(self.mod_sel)):
        #     print('--------------------------------------------')
        #     print("true",h_wave_mode_pol[i,0,self.ind_f[i]]/self.ref_wave_mode_pol[i,0,self.ind_f[i]])
        #     print("app",app_ratios[i,:])
        #     print("max", np.max(np.abs ( (bb[:,:,0]-bb[:,:,1])/bb[:,:,1] )) )

        
        # construct matrix
        Mat_F = [self.xp.array([[self.xp.ones_like(ff), ff, ff**2] for ff in fev]) for fev in self.f_to_eval]
        # get solutions for A and B vector
        r_vec = self.xp.array([[self.xp.linalg.solve(Mat_F[mod_numb], bb[mod_numb,:, pol]) for pol in range(2)] for mod_numb in range(len(self.mod_sel))])
        r_mat = self.xp.array([[
            [
                self.xp.conj(r_vec[v1, pol, 0])*r_vec[v2, pol, 0],
                self.xp.conj(r_vec[v1, pol, 0])*r_vec[v2, pol, 1] + self.xp.conj(r_vec[v1, pol, 1])*r_vec[v2, pol, 0],
                self.xp.conj(r_vec[v1, pol, 0])*r_vec[v2, pol, 2] + self.xp.conj(r_vec[v1, pol, 1])*r_vec[v2, pol, 1] + self.xp.conj(r_vec[v1, pol, 2])*r_vec[v2, pol, 0],
                self.xp.conj(r_vec[v1, pol, 1])*r_vec[v2, pol, 2] + self.xp.conj(r_vec[v1, pol, 2])*r_vec[v2, pol, 1],
                self.xp.conj(r_vec[v1, pol, 2])*r_vec[v2, pol, 2]
            ]  for v1 in range(len(self.mod_sel)) for v2 in range(len(self.mod_sel)) if v1<=v2] for pol in range(2)])


        d_h_app = self.xp.real(self.xp.sum(self.A_vec*r_vec))
        # breakpoint()
        h_h_app = self.xp.sum(self.xp.real(self.B_matrix_p*r_mat))#self.xp.sum(self.xp.real(self.xp.sum(self.B_matrix_p*r_mat, axis=1))) #self.xp.sum(xp.array([[self.xp.real(self.xp.dot(self.B_matrix_p[pol,:,mm],r_mat[pol,:,mm])) for mm in range(len(self.mod_sel))] for pol in range(2)]))

        like_out = -1.0 / 2.0 * (self.d_d + h_h_app - 2 * d_h_app).real
        print("d_h_app", d_h_app, "h_h_app", h_h_app, "like", like_out)

        # back to CPU if on GPU
        try:
            return like_out.get()

        except AttributeError:
            return like_out

    def A_vector(self, d, h_ref):
        freq = self.f_in
        return self.xp.array([4.0*self.xp.dot(self.xp.conj(d), h_ref*fpow/ self.psd) for fpow in [self.xp.ones_like(freq), freq, freq**2]])

    def B_vector(self, d, h_ref):
        freq = self.f_in
        return self.xp.array([4.0*self.xp.dot(self.xp.conj(d), h_ref*fpow/ self.psd)  for fpow in [self.xp.ones_like(freq), freq, freq**2, freq**3, freq**4]])

    def transform_to_fft_hp_hcross(self, wave):
        fd_sig = -self.xp.flip(wave)

        ind =int(( len(fd_sig) - 1 ) / 2 + 1)

        fft_sig_r = self.xp.real(fd_sig + self.xp.flip(fd_sig) )/2.0 + 1j * self.xp.imag(fd_sig - self.xp.flip(fd_sig))/2.0
        fft_sig_i = -self.xp.imag(fd_sig + self.xp.flip(fd_sig) )/2.0 + 1j * self.xp.real(fd_sig - self.xp.flip(fd_sig))/2.0
        return [fft_sig_r[ind:], fft_sig_i[ind:]]

    def PowerSpectralDensity(self, f):
        """
        PSD obtained from: https://arxiv.org/pdf/1803.01944.pdf
        """
        sky_averaging_constant = 1.0 # set to one for one source
        #(20/3) # Sky Averaged <--- I got this from Jonathan's notes
        L = 2.5*10**9   # Length of LISA arm
        f0 = 19.09*10**(-3)    

        Poms = ((1.5e-11)*(1.5e-11))*(1 + self.xp.power((2e-3)/f, 4))  # Optical Metrology Sensor
        Pacc = (3e-15)*(3e-15)* (1 + (4e-4/f)*(4e-4/f))*(1 + self.xp.power(f/(8e-3),4 ))  # Acceleration Noise
        alpha = 0.171
        beta = 292
        k =1020
        gamma = 1680
        f_k = 0.00215 
        Sc = 9e-45 * self.xp.power(f,-7/3)*self.xp.exp(-self.xp.power(f,alpha) + beta*f*self.xp.sin(k*f)) * (1 \
                                                + self.xp.tanh(gamma*(f_k- f)))  

        PSD = (sky_averaging_constant)* ((10/(3*L*L))*(Poms + (4*Pacc)/(self.xp.power(2*self.xp.pi*f,4)))*(1 + 0.6*(f/f0)*(f/f0)) + Sc) # PSD

        return PSD

    def InnerProduct(self, sig1, sig2):
        """Calculate the inner product.
        """
        
        if isinstance(sig1, list) is False:
            sig1 = [sig1]

        if isinstance(sig2, list) is False:
            sig2 = [sig2]
        
        if len(sig1) != len(sig2):
            raise ValueError(
                "Signal 1 has {} channels. Signal 2 has {} channels. Must be equal.".format(
                    len(sig1), len(sig2)
                )
            )
        
        out = 0.0
        for temp1, temp2 in zip(sig1, sig2):
            # get the lesser of the two lengths
            min_len = int(np.min([len(temp1), len(temp2)]))

            if len(temp1) != len(temp2):
                warnings.warn(
                    "The two time series are not the same length ({} vs {}). The calculation will run with length {} starting at index 0 for both arrays.".format(
                        len(temp1), len(temp2), min_len
                    )
                )

            # chop off excess length on a longer array
            # take fft
            temp1_fft = temp1[:min_len]
            temp2_fft = temp2[:min_len]

            # autocorrelation
            out += 4.0 * self.xp.dot(temp1_fft.conj(), temp2_fft/ self.psd) 

        # if using cupy, it will return a dimensionless array
        if self.use_gpu:
            return out.item().real
        return out.real

###############################################################################
sum_kwargs = dict(pad_output=True, output_type="fd")

fd_wave = GenerateEMRIWaveform(
    "FastSchwarzschildEccentricFlux",
    sum_kwargs=sum_kwargs,
    use_gpu=gpu_available,
    return_list=False,
)
T = 1.0  # years
dt = 10.0  # seconds

M = 1e6
mu = 5e1
p0 = 10.0
e0 = 0.3

dist = 4.10864264e00
Phi_phi0 = 1.4
Phi_r0 = 0.8
Phi_theta0 = 1.1

# define other parameters necessary for calculation
a = 0.0
Y0 = 1.0
qS = 0.5420879369091457
phiS = 5.3576560705195275
qK = 1.7348119514252445
phiK = 3.2004167279159637
############################################################################
from scipy.stats import uniform

def uniform_dist(min, max):
    if min > max:
        temp = min
        min = max
        max = temp

    mean = (max + min) / 2.0
    sig = max - min
    dist = uniform(min, sig)
    return dist

test_inds = np.array([0, 1, 3, 4]) 

# injection array
injection_params = np.array(
    [
        M,
        mu,
        a,  # will ignore
        p0,
        e0,
        Y0,  # will ignore
        dist,
        qS,
        phiS,
        qK,
        phiK,
        Phi_phi0,
        Phi_theta0,  # will ignore
        Phi_r0,
    ]
)

perc = 1e-2
priors = [uniform_dist(injection_params[test_inds[0]]*(1-perc), injection_params[test_inds[0]]*(1+perc)), 
             uniform_dist(injection_params[test_inds[1]]*(1-perc), injection_params[test_inds[1]]*(1+perc)),
             ]


N=int(3.14e6+1)
eps=5e-1
waveform_kwargs = {"T": T, "dt": dt, "eps": eps}#"mode_selection": [ (2,2,0), (2,2,1), (3,2,0), (2,2,-1)]}##


gen_wave = fd_waveform(fd_wave, N=N, use_gpu=gpu_available)
gen_wave.f_arr
gen_wave(*injection_params)

gen_wave.inject_signal(test_inds, *injection_params,**waveform_kwargs)

# gen_wave.get_RB_ll(injection_params[test_inds])
# gen_wave.get_ll(injection_params[test_inds] )
true_ll = gen_wave.get_ll(injection_params[test_inds])
app_ll = gen_wave.get_RB_ll(injection_params[test_inds])
print("ll true", true_ll, " app ll ", app_ll ) 

for i in range(50):
    print('-----------------------------------------------')
    factor = 10**(-np.random.randint(6,9))
    start_points = injection_params[test_inds].copy()
    start_points = injection_params[test_inds] * (1 + factor * np.random.normal( len(test_inds) ) )
    true_ll = gen_wave.get_ll(start_points)
    app_ll = gen_wave.get_RB_ll(start_points)
    print("\n")
    print("ll true", true_ll, " relative diff ", (true_ll - app_ll)/true_ll, " absolute diff ", (true_ll - app_ll) ) 
    


####################################

class Likelihood:
    def __init__(self, lnlike, priors, relative_binning=True):
        self.priors = priors
        self.lnlike = lnlike
        if relative_binning:
            self.loglike_func = self.lnlike.get_RB_ll
        else:
            self.loglike_func = self.lnlike.get_ll

    def __call__(self, x):
        prior_vals = np.zeros((x.shape[0]))
        for prior_i, x_i in zip(self.priors, x.T):
            temp = prior_i.logpdf(x_i)

            prior_vals[np.isinf(temp)] += -np.inf
            
        inds_eval = np.atleast_1d(np.squeeze(np.where(np.isinf(prior_vals) != True)))

        loglike_vals = np.full(x.shape[0], -np.inf)

        if len(inds_eval) == 0:
            return np.array([-loglike_vals, prior_vals]).T

        
        temp = [self.loglike_func(good_x) for good_x in x[inds_eval]]

        loglike_vals[inds_eval] = temp

        return np.array([loglike_vals, prior_vals]).T

like = Likelihood(gen_wave, priors)

ndim = len(test_inds)
nwalkers = 16

factor = 1e-5
start_points = injection_params[test_inds] * (1 + factor * np.random.randn(nwalkers, ndim))

print(like(start_points))

import emcee

sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            like,
            vectorize=True,
        )

nsteps = 1000
sampler.reset()
sampler.run_mcmc(start_points, nsteps, progress=True)