
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

folder = 'data_for_lorenzo/fluxes/*'
fluxfiles = [el for el in glob.glob(folder) if 'xI' in el]
# fluxfiles = [el for el in glob.glob(folder) if 'xI1' in el]

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

    # plt.figure()
    # plt.title('edot')
    # cb= plt.tricontourf(u,w, Ldot[-1])
    # plt.colorbar(cb)
    # plt.xlabel('u')
    # plt.ylabel('w')
    # plt.savefig(f"flux_check/{a[0]*xi[0]}_ldot.pdf")
    # plt.close()

flat_a = np.round(np.asarray(a_tot).flatten(),decimals=5)
flat_u = np.round(np.asarray(u_tot).flatten(),decimals=5)
flat_w = np.round(np.asarray(w_tot).flatten(),decimals=5)

flat_pdot = np.asarray(pdot).flatten()
flat_edot = np.asarray(edot).flatten()
flat_Edot = np.asarray(Edot).flatten()
flat_Ldot = np.asarray(Ldot).flatten()

def get_pdot(aa,uu,ww):
    mask = (aa==flat_a)*(uu==flat_u)*(ww==flat_w)
    return flat_pdot[mask][0]

def get_edot(aa,uu,ww):
    mask = (aa==flat_a)*(uu==flat_u)*(ww==flat_w)
    return flat_edot[mask][0]

def get_Edot(aa,uu,ww):
    mask = (aa==flat_a)*(uu==flat_u)*(ww==flat_w)
    return flat_Edot[mask][0]

def get_Ldot(aa,uu,ww):
    mask = (aa==flat_a)*(uu==flat_u)*(ww==flat_w)
    return flat_Ldot[mask][0]


a_unique = np.unique(flat_a)
u_unique = np.unique(flat_u)
w_unique = np.unique(flat_w)

x1 = a_unique.copy()
x2 = u_unique.copy()
x3 = w_unique.copy()
X = [x1, x2, x3]

for get,lab in zip([get_pdot,get_edot,get_Edot,get_Ldot], ['pdot', 'edot','Endot', 'Ldot']):
    reshapedF = np.asarray([[[get(el1,el2,el3) for el3 in x3] for el2 in x2] for el1 in x1])

    # flux interpolation
    InterpFlux = TPI.TP_Interpolant_ND(X, F=reshapedF)

    coeff = InterpFlux.GetSplineCoefficientsND().flatten()

    # np.savetxt(f'few/files/coeff_' + lab +'.dat', coeff)
    save_txt(coeff, f'few/files/coeff_' + lab +'.dat')

for i,el in enumerate(X):
    # print(el.shape)
    # np.savetxt(f'few/files/x{i}.dat', el)
    save_txt(el, f'few/files/x{i}.dat')

sepX = np.load("few/files/sepX.npy")
sepVals = np.load("few/files/sepVals.npy")

# flux interpolation
InterpSep = TPI.TP_Interpolant_ND([sepX[0], sepX[1]], F=sepVals)

coeff = InterpSep.GetSplineCoefficientsND().flatten()

save_txt(coeff, f'few/files/coeff_sep.dat')

for i,el in enumerate(sepX):
    save_txt(el, f'few/files/sep_x{i}.dat')