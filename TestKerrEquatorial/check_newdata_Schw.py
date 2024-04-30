import matplotlib.pyplot as plt
import numpy as np
import TPI
from few.trajectory.inspiral import EMRIInspiral
from few.utils.utility import get_overlap, get_mismatch, get_separatrix, get_fundamental_frequencies, get_fundamental_frequencies_spin_corrections
from scipy.interpolate import RegularGridInterpolator
from few.utils.constants import *
import time
traj = EMRIInspiral(func='SchwarzEccFlux')
import glob
a_tot, u_tot, w_tot = [], [], []

alpha = 4.0
deltap = 0.05
beta = alpha - deltap

folder = '../data_for_FEW/fluxes/*'
fluxfiles = [el for el in glob.glob(folder) if 'a0.00' in el]
M,mu=1e6,10
folder = './KerrEquatoriaChecks'
for ff in fluxfiles:
    imp = np.loadtxt(ff)
    a, p, e, xi, E, Lz, Q, pLSO, EdotInf_tot, EdotH_tot, LzdotInf_tot, LzdotH_tot, QdotInf_tot, QdotH_tot, pdotInf_tot, pdotH_tot, eccdotInf_tot, eccdotH_tot, xidotInf_tot, xidotH_tot = imp.T

    diff_sep = np.asarray([pLSO[ii] - get_separatrix(a[ii],e[ii],xi[ii]) for ii in range(e.shape[0])])
    u = np.log((p-pLSO + beta)/alpha)
    w = np.sqrt(e)
    for el in [u,w]:
        delta_var = np.diff(el)
        step = delta_var[delta_var!=0.0][0]
        var_min, var_max = el.min(), el.max()
        num = int((var_max - var_min)/step)
    rhs = np.asarray([traj.get_rhs_ode(M,mu,aa,pp,ee,np.sign(xx))/(mu/M) for aa,pp,ee,xx in zip(a,p,e,xi)])
    diff_pdot = rhs[:,0]-(pdotInf_tot+pdotH_tot)
    diff_eccdot = rhs[:,1]-(eccdotInf_tot+eccdotH_tot)

    mask = (e>-1.0)#(e<0.6)*(p<8.0)*(e>0.0)
    
    dlne_dlnp =(eccdotInf_tot+eccdotH_tot)/(pdotInf_tot+pdotH_tot) * p/e
    
    omega_phi = np.asarray([get_fundamental_frequencies(aa, pp, ee, xx)[0] for aa,pp,ee,xx in zip(a,p,e,xi)])
    dlnE_dlnL = (EdotInf_tot + EdotH_tot)/ (LzdotInf_tot + LzdotH_tot) / omega_phi # * Lz/E 
    
    dOmega_dp =omega_phi/(pdotInf_tot+pdotH_tot)
    dOmega_decc =omega_phi/(eccdotInf_tot+eccdotH_tot)
    
    # mask = (e>0.0)*(p<8.0)
    # plt.figure()
    # plt.title('Vector field Trajectory')
    # cb= plt.tricontourf(p[mask], e[mask], (dlne_dlnp)[mask]  )
    # plt.colorbar(cb,label=r'$\frac{d \ln e}{ d \ln p}$')
    # plt.xlabel('p')
    # plt.ylabel('e')
    # plt.tight_layout()
    # plt.show()
    
    # plt.figure()
    # plt.title('Vector field Trajectory')
    # cb= plt.tricontourf(p[mask], e[mask], (dOmega_dp)[mask]  )
    # plt.colorbar(cb,label=r'$\frac{d \Omega}{ d p}$')
    # plt.xlabel('p')
    # plt.ylabel('e')
    # plt.tight_layout()
    # plt.show()
    
    # plt.figure()
    # plt.title('Vector field Trajectory')
    # cb= plt.tricontourf(p[mask], e[mask], (dOmega_decc)[mask]  )
    # plt.colorbar(cb,label=r'$\frac{d \Omega}{ d e}$')
    # plt.xlabel('p')
    # plt.ylabel('e')
    # plt.tight_layout()
    # plt.show()
    
    mask = (e>0.0)
    # plt.figure()
    # plt.title('Vector field trajectory')
    # cb= plt.tricontourf(p[mask], e[mask], (dlnE_dlnL)[mask]  )
    # plt.colorbar(cb,label=r'$\frac{\dot {E} }{\Omega_\Phi \, \dot{L}}$')
    # plt.xlabel('p')
    # plt.ylabel('e')
    # plt.tight_layout()
    # plt.show()
    # plt.savefig('dLnEcc_dLnP.png')
    
    plt.figure()
    plt.title('Difference between SchwarzEccFlux interp and Scott data')
    cb= plt.tricontourf(p[mask], e[mask], np.log10(np.abs(diff_pdot))[mask]  )
    plt.colorbar(cb,label='log10 error pdot')
    plt.xlabel('p')
    plt.ylabel('e')
    plt.tight_layout()
    plt.savefig('diff_pdot_schwVSnewdata.png')
    
    plt.figure()
    plt.title('Difference between SchwarzEccFlux interp and Scott data')
    cb= plt.tricontourf(p[mask], e[mask], np.log10(np.abs(diff_eccdot))[mask]  )
    plt.colorbar(cb,label='log10 error eccdot')
    plt.xlabel('p')
    plt.ylabel('e')
    plt.tight_layout()
    plt.savefig('diff_eccdot_schwVSnewdata.png')
    
    plt.figure(); 
    plt.semilogy(e[mask], -rhs[:,0][mask],'.',label='SchwarzInterp2d'); 
    plt.semilogy(e[mask], -(pdotInf_tot+pdotH_tot)[mask],'x',label='KerrData'); 
    plt.xlabel('eccentricity')
    plt.ylabel('pdot')
    plt.legend()
    plt.savefig('pdot_ecc_schwVSnewdata.png')
    
    plt.figure(); 
    plt.semilogy(e[mask], -rhs[:,1][mask],'.',label='SchwarzInterp2d'); 
    plt.semilogy(e[mask], -(eccdotInf_tot+eccdotH_tot)[mask],'x',label='KerrData'); 
    plt.xlabel('eccentricity')
    plt.ylabel('eccdot')
    plt.legend()
    plt.savefig('eccdot_ecc_schwVSnewdata.png')

