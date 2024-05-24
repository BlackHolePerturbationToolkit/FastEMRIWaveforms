#python -m unittest few/tests/test_traj.py 
import unittest
import numpy as np
import warnings
from scipy.interpolate import CubicSpline
import time
import matplotlib.pyplot as plt

from few.trajectory.inspiral import EMRIInspiral
from few.amplitude.romannet import RomanAmplitude
from few.amplitude.interp2dcubicspline import Interp2DAmplitude
from few.waveform import FastSchwarzschildEccentricFlux, SlowSchwarzschildEccentricFlux
from few.utils.utility import get_overlap, get_mismatch, get_separatrix, get_fundamental_frequencies
from few.utils.ylm import GetYlms
from few.utils.modeselector import ModeSelector
from few.summation.interpolatedmodesum import CubicSplineInterpolant
from few.utils.constants import *

try:
    import cupy as xp

    gpu_available = True

except (ModuleNotFoundError, ImportError) as e:
    import numpy as xp

    warnings.warn(
        "CuPy is not installed or a gpu is not available. If trying to run on a gpu, please install CuPy."
    )
    gpu_available = False

T = 100.0
dt = 10.0

insp_kw = {
"T": T,
"dt": dt,
"err": 1e-10,
"DENSE_STEPPING": 0,
"max_init_len": int(1e4),
"use_rk4": False,
"upsample": False,
}

np.random.seed(42)

class ModuleTest(unittest.TestCase):
    def test_trajectory_pn5(self):

        # initialize trajectory class
        traj = EMRIInspiral(func="pn5")

        # set initial parameters
        M = 1e5
        mu = 1e1
        np.random.seed(42)
        for i in range(10):
            p0 = np.random.uniform(10.0,15)
            e0 = np.random.uniform(0.0, 1.0)
            a = np.random.uniform(0.0, 1.0)
            Y0 = np.random.uniform(-1.0, 1.0)

            # do not want to be too close to polar
            if np.abs(Y0) < 1e-2:
                Y0 = np.sign(Y0) * 1e-2

            # run trajectory
            #print("start", a, p0, e0, Y0)
            t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, Y0, **insp_kw)

    def test_trajectory_SchwarzEccFlux(self):
        # initialize trajectory class
        traj = EMRIInspiral(func="SchwarzEccFlux")

        # set initial parameters
        M = 1e5
        mu = 1e1
        p0 = 10.0
        e0 = 0.7

        # run trajectory
        t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, 0.0, p0, e0, 1.0)

    def test_trajectory_KerrEccentricEquatorial(self):

        err = 1e-10
        
        # initialize trajectory class
        list_func = ['KerrEccentricEquatorial_ELQ_nofrequencies', 'KerrEccentricEquatorial', 'KerrEccentricEquatorial_nofrequencies', 'KerrEccentricEquatorial_ELQ', ]
        # list_func = ['KerrEccentricEquatorial','KerrEccentricEquatorial_ELQ', ]
        for el in list_func:
            print("testing ", el)
            traj = EMRIInspiral(func=el)

            # set initial parameters
            M = 1e6
            mu = 100.0
            
            if 'nofrequencies' in el:
                insp_kw["use_rk4"] = True
            else:
                insp_kw["use_rk4"] = False
            
            # plt.figure()
            Np = 0
            tic = time.perf_counter()
            for i in range(100):
                
                p0 = np.random.uniform(9.0,12.0)
                e0 = np.random.uniform(0.01, 0.1)
                a = np.random.uniform(0.01, 0.98)
                # print(a,p0,e0)
                # run trajectory        
                t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, np.abs(a), p0, e0, np.sign(a)*1.0, **insp_kw)
                # plt.plot(p,e,'.',label=f'N={len(t)}',alpha=0.7)
                # print(a,p0,e0,len(t))
                Np += len(t)
                
            toc = time.perf_counter()
            print('timing trajectory ',(toc-tic)/100, "Np=",Np/100)
        
            # plt.legend(); plt.xlabel('p'); plt.ylabel('e'); plt.tight_layout()
            # plt.savefig(el + f'a_p_e_1e-16.png')

        # test against Schwarz
        traj_Schw = EMRIInspiral(func="SchwarzEccFlux")
        a=0.0
        charge = 0.0

        # check against Schwarzchild
        for i in range(100):
            p0 = np.random.uniform(10.0,15)
            e0 = np.random.uniform(0.1, 0.5)
            
            t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, 1.0, T=2.0, max_init_len=int(1e5))
            tS, pS, eS, xS, Phi_phiS, Phi_thetaS, Phi_rS = traj_Schw(M, mu, 0.0, p0, e0, 1.0, T=2.0, new_t=t, upsample=True, max_init_len=int(1e5))
            mask = (Phi_rS!=0.0)
            diff =  np.abs(Phi_phi[mask] - Phi_phiS[mask])
            # plt.figure(); plt.plot(tS,pS);plt.plot(t,p);plt.show()
            # plt.figure(); plt.plot(tS,Phi_phiS);plt.plot(t,Phi_phi);plt.show()

            self.assertLess(np.max(diff),2.0)
        
        t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(1e6, 100.0, 0.99, 6.0, 0.5, 1.0, T=2079375.6399400292/YRSID_SI)
        self.assertLess(np.abs(Phi_phi[-1] - 37548.68909110543),2.0) # value from Scott
        
        # s_t, s_p, s_e, s_x, s_omr, s_omt, s_omph, s_r, s_th, s_ph = np.loadtxt("data_for_lorenzo/scott_data/a0.99_p0_6_e0_0.5_xI0_1.0_wl.txt").T
        # mask = (s_p>(0.1+s_p[-1]))
        # t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(1e6, 100.0, 0.99, s_p[0], s_e[0], 1.0, T=4.0, new_t=s_t[mask]*MTSUN_SI*M, upsample=True)
        # # plt.figure();  plt.plot(t,p); plt.plot(s_t*MTSUN_SI*M,s_p,'--',label='Scott'); plt.show()
        # plt.figure();  plt.plot(p,e); plt.plot(s_p[mask],s_e[mask],'--',label='Scott'); plt.show()
        # plt.figure();  plt.semilogy(t,np.abs(Phi_phi-s_ph[mask])); plt.ylabel('phase difference phi'); plt.xlabel('time [seconds]'); plt.show()