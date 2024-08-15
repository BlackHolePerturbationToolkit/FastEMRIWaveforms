
import matplotlib.pyplot as plt
import numpy as np

# install TPI: pip install tpi-splines
import TPI
from few.utils.spline import CubicSpline, BicubicSpline, TricubicSpline

from few.trajectory.inspiral import EMRIInspiral
from few.utils.utility import get_overlap, get_mismatch, get_separatrix, get_fundamental_frequencies, get_fundamental_frequencies_spin_corrections
from scipy.interpolate import RegularGridInterpolator

def save_txt(my_array, fname):
    # Open the file in write mode
    with open(fname, "w") as file:
        # Write each element of the array to a new line in the file
        for element in my_array:
            file.write("{:.16f}".format(element) + "\n")

trajpn5 = EMRIInspiral(func="pn5")
trajS = EMRIInspiral(func="SchwarzEccFlux")

filepath = '../../data_for_FEW/fluxes/a{:.2f}_xI{:.3f}.flux'

Nx1 = 128
Nx2 = 128
amin = -0.99998
amax = 0.99998
emin,emax = 0.0, 0.9

def a_to_chi2(a):
    scale=3.
    y = (1-a)**(1/scale)
    ymin = (1-amax)**(1/scale)
    ymax = (1+amax)**(1/scale)
    return (y-ymin)/(ymax-ymin)

def chi2_to_a(chi2):
    scale=3.
    ymin = (1-amax)**(1/scale)
    ymax = (1+amax)**(1/scale)
    return 1-(chi2*(ymax-ymin)+ymin)**scale

x1 = np.linspace(0,1,num=Nx1)
x2 = np.linspace(emin**0.5,emax**0.5,num=Nx2)

chi2, sqrtecc = np.meshgrid(x1,x2, indexing='ij')

X = [x1,x2]

spin = chi2_to_a(chi2.flatten())
e = sqrtecc.flatten()**2
to_interp = get_separatrix(np.abs(spin), e, np.sign(spin)*1.0)/(6.+2.*e)

reshapedF = np.asarray(to_interp).reshape((Nx1, Nx2))

save_txt(reshapedF.flatten(), f'../../few/files/TricubicData_psep.dat')

for i,el in enumerate(X):
    save_txt(el, f'../../few/files/TricubicData_psep_x{i}.dat')

for bc in ["E(3)","not-a-knot"]: #"natural","not-a-knot","clamped", "E(3)", "natural-alt"
    interpTR = BicubicSpline(x1, x2, reshapedF, bc=bc)
    interpTP = TPI.TP_Interpolant_ND(X, F=reshapedF)
    
    # test the interpolation against each other
    Ntest=100
    # x1_test = np.random.uniform(0, 1, Ntest)
    # x2_test = np.random.uniform(emin**0.5, emax**0.5, Ntest)
    x1_vec = np.linspace(0.0001,0.9999,Ntest)
    x2_vec = np.linspace((emin+1e-5)**0.5, (emax-1e-5)**0.5, Ntest)

    mesh = np.meshgrid(x1_vec, x2_vec)
    x1_test = mesh[0].flatten()
    x2_test = mesh[1].flatten()

    eval_TP = np.array([interpTP((x1_test[ii], x2_test[ii],)) for ii in range(Ntest**2)])
    eval_TR = np.array([interpTR(x1_test[ii], x2_test[ii],) for ii in range(Ntest**2)])
    difference_interpolants = eval_TR - eval_TP

    a_transformed = chi2_to_a(x1_test)
    e_in = x2_test**2
    a_in = abs(a_transformed)
    xi_in = np.sign(a_transformed) * 1.

    truth = get_separatrix(a_in, e_in, xi_in)
    
    TR_vs_truth = eval_TR * (6 + 2*e_in) - truth
    TP_vs_truth = eval_TP * (6 + 2*e_in) - truth

    print(a_to_chi2(-0.99998))

    # # make a surface plot along x2,x3 axis of the error
    # if bc != "not-a-knot":
    #     plt.figure()
    #     plt.tricontourf(x1_test, x2_test, np.log10(np.abs(difference_interpolants)))
    #     plt.colorbar(label='log10 error')
    #     plt.title(f"Separatrix")
    #     plt.ylabel('w=sqrt(e)')
    #     plt.xlabel('chi2 (0 = amax)')
    #     plt.savefig(f"./psep_TPvsTR"+bc+"_error.png")

    plt.figure()
    plt.tricontourf(x1_test, x2_test, np.log10(np.abs(TR_vs_truth)))
    plt.colorbar(label='log10 error')
    plt.title(f"Separatrix")
    plt.ylabel('w=sqrt(e)')
    plt.xlabel('chi2 (0 = amax)')
    plt.savefig(f"./psep_TRvsTruth"+bc+"_error.png")

    plt.figure()
    plt.tricontourf(x1_test, x2_test, np.log10(np.abs(TP_vs_truth)))
    plt.colorbar(label='log10 error')
    plt.title(f"Separatrix")
    plt.ylabel('w=sqrt(e)')
    plt.xlabel('chi2 (0 = amax)')
    plt.savefig(f"./psep_TPvsTruth"+bc+"_error.png")
