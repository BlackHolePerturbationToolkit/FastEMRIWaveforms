import numpy as np
import warnings
import matplotlib.pyplot as plt
try:
    import cupy as xp

    gpu_available = True

except (ModuleNotFoundError, ImportError) as e:
    import numpy as xp

    warnings.warn(
        "CuPy is not installed or a gpu is not available. If trying to run on a gpu, please install CuPy."
    )
    gpu_available = False


from few.trajectory.inspiral import EMRIInspiral
from few.utils.baseclasses import SchwarzschildEccentric, Pn5AAK, ParallelModuleBase
from few.waveform import AAKWaveformBase
from few.summation.aakwave import AAKSummation

insp_kwargs = {
    "err": 1e-10,
    "DENSE_STEPPING": 0,
    "max_init_len": int(1e4),
    "func":"KerrEccentricEquatorial"
    }

sum_kwargs = dict(use_gpu=gpu_available)
num_threads = 16

waveform_class = AAKWaveformBase(
            EMRIInspiral,
            AAKSummation,
            inspiral_kwargs=insp_kwargs,
            sum_kwargs=sum_kwargs,
            use_gpu=gpu_available,
            num_threads=num_threads,
        )

####################
# set initial parameters
M = 5e5
mu = 1e1
a = 0.85
p0 = 8.0
e0 = 0.4
Y0 = 1.0
Phi_phi0 = 0.2
Phi_theta0 = 1.2
Phi_r0 = 0.8

qS = 0.2
phiS = 0.2
qK = 0.8
phiK = 0.8
dist = 1.0
mich = False
dt = 1.0
T = 1.0

# new input
secondary_spin = 1e-4

# plot as a function of dt
dtvec = np.linspace(1,15, num=2)
plt.figure()
for dt in dtvec:
    output = waveform_class(M,
        mu,
        a,
        p0,
        e0,
        Y0,
        dist,
        qS,
        phiS,
        qK,
        phiK,
        secondary_spin,
        Phi_phi0=0.0,
        Phi_theta0=0.0,
        Phi_r0=0.0,
        mich=False,
        dt=dt,
        T=T
        )

    time = np.arange(len(output))*dt
    plt.plot(time[-100:], output.real[-100:],label=f'{dt}')
plt.legend()
plt.show()