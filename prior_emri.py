import time
import numpy as np
import matplotlib.pyplot as plt
from few.utils.utility import *
from few.utils.constants import *
from few.trajectory.inspiral import EMRIInspiral
from few.amplitude.romannet import RomanAmplitude
from scipy.interpolate import CubicSpline
from few.utils.ylm import GetYlms

from few.summation.interpolatedmodesum import CubicSplineInterpolant
from multiprocessing import Pool


# amplitudes
amp = RomanAmplitude()

func = "SchwarzEccFlux"
traj = EMRIInspiral(func=func)


keys = np.asarray(list(amp.index_map.keys()))
values = np.asarray(list(amp.index_map.values()))

mn_dict = {}
tot_check = 0
for mm in range(-10,11):
    for nn in range(-30,31):
        if ((mm==0)and(nn==0)):
            mask = (keys[:,1:] == [mm,nn])
            print(mm,nn,values[np.prod(mask,axis=1,dtype=bool)])
        else:
            mask = (keys[:,1:] == [mm,nn])
            mn_dict[(mm,nn)] = values[np.prod(mask,axis=1,dtype=bool)]
            tot_check += len(mn_dict[(mm,nn)])

print(tot_check, len(values))
print("total harmonics", len(list(mn_dict.keys())))

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

def get_factor(t, freqs):
    cs = CubicSplineInterpolant(t, freqs)
    fdot = cs(t,deriv_order=1)
    fddot = cs(t,deriv_order=2)
    arg_1 = 2*np.pi*fdot**3 / (3*fddot**2)
    arg_1_cpu = arg_1
    spa_func = np.asarray(SPAFunc(arg_1_cpu))
    return -1*fdot/np.abs(fddot) * spa_func * 2./np.sqrt(3) / np.sqrt(arg_1+0j)

S_git = np.genfromtxt('../EMRI_FourierDomainWaveforms/LISA_Alloc_Sh.txt')
sensitivity_fn = CubicSpline(S_git[:,0], S_git[:,1])

def get_p0(Tobs, M, mu, e0):
    try:
        # fix p0 given T
        p0 = get_p_at_t(
        traj,
        Tobs,
        [M, mu, 0.0, e0, 1.0],
        index_of_p=3,
        index_of_a=2,
        index_of_e=4,
        index_of_x=5,
        traj_kwargs={},
        xtol=2e-12,
        rtol=8.881784197001252e-16,
        bounds=[6+2*e0+0.1, 16.0],)
    except:
        return 0.0
    return p0

def get_approx_SNR(M, mu, p0, e0, T, dist=1.0):
    if p0 == 0.0:
        return 0.0
    # get trajectory

    (t, p, e, x, Phi_phi, Phi_theta, Phi_r) = traj(M, mu, 0.0, p0, e0, 1.0, T=T, dt=10.0)
    m0mask = amp.m0mask != 0
    m0mask = m0mask
    num_m_zero_up = len(m0mask)
    num_m_1_up = len(np.arange(len(m0mask))[m0mask])
    num_m0 = len(np.arange(len(m0mask))[~m0mask])
    modeinds = [amp.l_arr, amp.m_arr, amp.n_arr]

    # amplitudes
    teuk_modes = amp(p, e, amp.l_arr, amp.m_arr, amp.n_arr)

    ylms = 1.0
    Msec = M * MTSUN_SI

    # get dimensionless fundamental frequency
    OmegaPhi, OmegaTheta, OmegaR = get_fundamental_frequencies(0.0, p, e, x)

    # get frequencies in Hz
    f_Phi, f_omega, f_r = OmegaPhi, OmegaTheta, OmegaR = (
        np.asarray(OmegaPhi) / (Msec * 2 * PI),
        np.asarray(OmegaTheta) / (Msec * 2 * PI),
        np.asarray(OmegaR) / (Msec * 2 * PI),
    )

    zero_modes_mask = (modeinds[1]==0)*(modeinds[2]==0)
    freqs = (
        modeinds[1][np.newaxis, :] * f_Phi[:, np.newaxis]
        + modeinds[2][np.newaxis, :] * f_r[:, np.newaxis]
    )

    freqs_shape = freqs.shape

    # make all frequencies positive
    freqs_in = np.abs(freqs)
    PSD = sensitivity_fn(freqs_in.flatten()).reshape(freqs_shape)

    # weight by PSD, only for non zero modes
    fact = get_factor(t, freqs.T).T
    ylm_gen = GetYlms(assume_positive_m=True, use_gpu=False)
    theta = np.pi/3.
    phi = np.pi/2.

    ylms = ylm_gen(amp.unique_l, amp.unique_m, theta, phi).copy()[amp.inverse_lm]

    Amp2 = np.concatenate([teuk_modes, np.conj(teuk_modes[:, m0mask])], axis=1) * ylms
    power = (2 * np.abs(Amp2 * fact)**2 /PSD)
    snr_harmonic = np.trapz(power, x=freqs, axis=0)

    # plot power
    # plt.figure(); plt.plot(freqs[:,amp.index_map[(2,2,0)]],power[:,amp.index_map[(2,2,0)]]); plt.show()
    inn_prod = np.sum(snr_harmonic[~zero_modes_mask])
    dist_dimensionless = (dist * Gpc) / (mu * MRSUN_SI)
    snr_out = np.sqrt(inn_prod) / dist_dimensionless
    # print(snr_out)
    return snr_out

Tobs = 2.0
mu = np.linspace(1,100,10)
e0 = 0.1
M = 10**np.linspace(5, 7, 10)

# create a contourf for SNR as a function of M and mu
plt.figure()
# time the calculation
tic = time.time()
def evaluate_SNR(mass, secmass):
    p0 = get_p0(Tobs, mass, secmass, e0)
    snr = get_approx_SNR(mass, secmass, p0, e0, Tobs)
    return snr

input = [(mass, secmass) for mass in M for secmass in mu]
# breakpoint()
# SNR = np.zeros((len(mu), len(M)))
# for i, (mass, secmass) in enumerate(input):
#     SNR[i // len(M), i % len(M)] = evaluate_SNR(mass, secmass)

with Pool(16) as pool:
    SNR = np.asarray(pool.starmap(evaluate_SNR, input))
    SNR = SNR.reshape((len(mu), len(M)))

toc = time.time()
print("Time taken: ", (toc-tic)/(M.shape[0]*mu.shape[0]))
print(SNR.max())
fig = plt.figure()
ax = fig.add_subplot(111)
contour = ax.contourf(M, mu, SNR, cmap='viridis')
ax.set_xlabel("M")
ax.set_ylabel("mu")
# log scale on M
ax.set_xscale('log')
# ax.set_yscale('log')
plt.colorbar(contour)
plt.savefig('SNR_contour_plot.png')
