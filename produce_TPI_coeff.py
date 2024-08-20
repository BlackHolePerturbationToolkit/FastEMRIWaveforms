
import matplotlib.pyplot as plt
import numpy as np
# install TPI: pip install tpi-splines
import TPI
from few.trajectory.inspiral import EMRIInspiral
from few.utils.utility import get_overlap, get_mismatch, get_separatrix, get_fundamental_frequencies, get_fundamental_frequencies_spin_corrections
from scipy.interpolate import RegularGridInterpolator
import glob

def save_txt(my_array, fname):
    # Open the file in write mode
    with open(fname, "w") as file:
        # Write each element of the array to a new line in the file
        for element in my_array:
            file.write("{:.16f}".format(element) + "\n")

def read_txt(fname):
    # Open the file in read mode
    with open(fname, "r") as file:
        # Read the contents of the file into a list
        lines = file.readlines()

    # Initialize an empty two-dimensional list
    data = []

    # Loop through each line in the file
    for line in lines:
        # Split the line into a list of values using a comma delimiter
        values = line.strip().split(" ")

        # Convert each value to a float and append to the two-dimensional list
        # data.append([np.float128(value) for value in values])
        data.append([np.float64(value) for value in values])
    
    return np.asarray(data)

def Edotpn(a, p, e, pLSO):
    """
    https://arxiv.org/pdf/2201.07044.pdf
    eq 91
    """
    pdot_V = 32./5. * p**(-5) * (1-e**2)**1.5 * (1 + 73/24 * e**2 + 37/96 * e**4)
    return pdot_V

def Ldotpn(a, p, e, pLSO):
    """
    https://arxiv.org/pdf/2201.07044.pdf
    eq 91
    """
    pdot_V = 32./5. * p**(-7/2) * (1-e**2)**1.5 * (1 + 7./8. * e**2)
    return pdot_V

def pdotpn(a, p, e, pLSO):
    """
    https://arxiv.org/pdf/2201.07044.pdf
    eq 91
    """
    risco = get_separatrix(a+1e-500, np.zeros_like(a), np.ones_like(a))
    U2 = (p - risco)**2 - (pLSO - risco)**2
    pdot_V = 8./5. * p**(-3) * (1-e**2)**1.5 * (8 + 7 * e**2) * (p**2 / U2)
    return pdot_V

def edotpn(a, p, e, pLSO):
    """
    https://arxiv.org/pdf/2201.07044.pdf
    eq 91
    without the factor of e
    """
    risco = get_separatrix(a+1e-500, np.zeros_like(a), np.ones_like(a))
    U2 = (p - risco)**2 - (pLSO - risco)**2
    return 1/15 * p**(-4) * (1-e**2)**1.5 * (304 + 121 * e**2) * (p**2 / U2) #* (e + 1e-500)

trajpn5 = EMRIInspiral(func="pn5")
trajS = EMRIInspiral(func="SchwarzEccFlux")

import glob
a_tot, u_tot, w_tot = [], [], []
pdot = []
edot = []
Ldot = []
Edot = []
plso = []

alpha = 4.0
deltap = 0.05
beta = alpha - deltap

filepath = 'data_for_FEW/fluxes/a{:.2f}_xI{:.3f}.flux'

avals = np.r_[np.linspace(0.,0.9,10),0.95,0.99]
avals = np.r_[-np.flip(avals)[:-1],avals]

a_in = abs(avals)
xi_in = np.sign(avals)*1.
xi_in[xi_in ==0] = 1

fluxfiles = [filepath.format(ah, xh) for ah, xh in zip(a_in, xi_in)]

for ff in fluxfiles:
    imp = read_txt(ff)
    a, p, e, xi, E, Lz, Q, pLSO, EdotInf_tot, EdotH_tot, LzdotInf_tot, LzdotH_tot, QdotInf_tot, QdotH_tot, pdotInf_tot, pdotH_tot, eccdotInf_tot, eccdotH_tot, xidotInf_tot, xidotH_tot = imp.T
    
    u = np.log((p-pLSO + beta)/alpha)
    w = np.sqrt(e)
    a_tot.append(a*xi )
    u_tot.append(u )
    w_tot.append(w )
    pdot.append( (pdotInf_tot+pdotH_tot ) / pdotpn(a, p, e, pLSO) )
    edot.append( (eccdotInf_tot+eccdotH_tot) / edotpn(a, p, e, pLSO) )
    Edot.append( (EdotInf_tot+EdotH_tot) / Edotpn(a, p, e, pLSO) )
    Ldot.append( np.abs(LzdotInf_tot+LzdotH_tot) / Ldotpn(a, p, e, pLSO) )
    plso.append(pLSO )

# get grid from file and construct with meshgrid if any checking needed
_, _, _, _, grid_u, _, grid_w = np.loadtxt("data_for_FEW/fluxes/flux.grid").T
unique_u_grid = np.flip(np.unique(grid_u))
unique_w_grid = np.unique(grid_w)
unique_a_grid = avals

flat_a, flat_w, flat_u = np.meshgrid(unique_a_grid,unique_w_grid, unique_u_grid, indexing='ij')

x1 = avals.copy()
x2 = np.unique(flat_w)
x3 = np.unique(flat_u)
X = [x1, x2, x3]

for flux, lab in zip([pdot, edot, Edot, Ldot], ['pdot', 'edot','Endot', 'Ldot']):
    reshapedF = np.asarray(flux).reshape(flat_u.shape)
    reshapedF = np.flip(reshapedF, axis=2)  # u must be increasing
    # flux interpolation
    lower_bcs = ["not-a-knot","clamped","not-a-knot"]
    upper_bcs = ["not-a-knot","not-a-knot","not-a-knot"]
    InterpFlux = TPI.TP_Interpolant_ND(X, F=reshapedF, lower_bcs=lower_bcs, upper_bcs=upper_bcs)

    coeff = InterpFlux.GetSplineCoefficientsND().flatten()

    # np.savetxt(f'few/files/coeff_' + lab +'.dat', coeff)
    save_txt(coeff, f'few/files/coeff_' + lab +'.dat')

for i,el in enumerate(X):
    # print(el.shape)
    # np.savetxt(f'few/files/x{i}.dat', el)
    save_txt(el, f'few/files/x{i}.dat')

# sepX = np.load("few/files/sepX.npy")
# sepVals = np.load("few/files/sepVals.npy")

# flux interpolation
# InterpSep = TPI.TP_Interpolant_ND([sepX[0], sepX[1]], F=sepVals, lower_bcs=["not-a-knot","clamped"])

# coeff = InterpSep.GetSplineCoefficientsND().flatten()

# save_txt(coeff, f'few/files/coeff_sep.dat')

# for i,el in enumerate(sepX):
#     save_txt(el, f'few/files/sep_x{i}.dat')