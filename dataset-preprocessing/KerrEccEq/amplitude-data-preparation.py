from few.amplitude.ampinterp2d import AmpInterpKerrEqEcc
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import h5py

# FEW expects coefficients in the same form as RectBivariateSpline, so we'll use it to construct things for us.
from scipy.interpolate import RectBivariateSpline

# fill all lmn mode values
md = []
for l in  range(2, 11):
    for m in range(0, l + 1):
        for n in range(-55, 56):
            md.append([l, m, n])

# total number of modes in the model
num_teuk_modes = len(md)

# mask for m == 0
m0mask = np.array(
    [
        m == 0
        for l in range(2, 11)
        for m in range(0, l + 1)
        for n in range(-55, 56)
    ]
)

# sorts so that order is m=0, m<0, m>0
m0sort = m0sort = np.concatenate(
    [
        np.arange(num_teuk_modes)[m0mask],
        np.arange(num_teuk_modes)[~m0mask],
    ]
)

# sorts the mode indexes
md = np.asarray(md).T[:, m0sort].astype(np.int32)

# store l m and n values
l_arr_no_mask = md[0]
m_arr_no_mask = md[1]
n_arr_no_mask = md[2]

data_loc = "./raw_data"

data_dir_A = os.path.join(data_loc, 'regionA')
data_dir_B = os.path.join(data_loc, 'regionB')

w_knots_inner = np.linspace(0,1,33)
u_knots_inner = np.linspace(0,1,33)

##### Quick explainer of what is going on below:
# We are creating a new HDF5 file to store the amplitude data. We will store the coefficients for each mode in the same format as RectBivariateSpline.
# We go mode by mode and plug them into RectBivariateSpline to get the coefficients.
#
# The convention of the raw amplitude data is to store +/-m and infer -n by symmetry, but FEW assumes that the data is missing -m instead.
# We therefore handle this symmetry flip when it is required.
# 
# RegionA is at full density (33 x 33 x 33) and RegionB is at half density (17 x 17 x 17). This is to save on file space.
# It is generally assumed that regionB does not need to be interpolated as accurately as regionA, but this can be assessed at a later stage.
#
# We also store the parameter grid for each region for completeness. 

with h5py.File("ZNAmps_l10_m10_n55_DS2Outer.h5", "w") as f:
    # REGION A
    params_path = os.path.join(data_dir_A, 'params.feather')
    paramsA = pd.read_feather(params_path)
    params = paramsA.to_numpy().reshape(33, 33, 33, 3)

    # lmax, mmax, nmax assumed the same for both regions
    f.attrs['lmax'] = 10
    f.attrs['mmax'] = 10
    f.attrs['nmax'] = 55

    regionA = f.create_group('regionA')
    coeff_arr_out = regionA.create_dataset("CoeffsRegionA", (33, l_arr_no_mask.size, 2, 33 * 33), dtype='f8')

    regionA.attrs['NZ'] = 33
    regionA.attrs['NW'] = 33
    regionA.attrs['NU'] = 33
    
    for k, (ell_select, emm_select, enn_select) in tqdm(enumerate(zip(l_arr_no_mask, m_arr_no_mask, n_arr_no_mask))):
        if enn_select >= 0:
            # print(ell_select, emm_select, enn_select)
            mode_path = os.path.join(data_dir_A, f'l{ell_select}', f'm{emm_select}', f'amplitudes_{ell_select}_{emm_select}_{enn_select}.feather')
            modetest = pd.read_feather(mode_path)
            mode = modetest.to_numpy().reshape(33, 33, 33, 2)
            
        else:
            emm_select = -emm_select
            enn_select = -enn_select
            # print("Flip:", ell_select, emm_select, enn_select)
            mode_path = os.path.join(data_dir_A, f'l{ell_select}', f'm{emm_select}', f'amplitudes_{ell_select}_{emm_select}_{enn_select}.feather')
            modetest = pd.read_feather(mode_path)
            mode = modetest.to_numpy().reshape(33, 33, 33, 2)
            mode[:,:,:,1] *= -1
            mode *= (-1) ** ell_select

        for spin_ind in range(33):
            spl1 = RectBivariateSpline(w_knots_inner, u_knots_inner, mode[:, :, spin_ind, 0].T, kx=3, ky=3)
            spl2 = RectBivariateSpline(w_knots_inner, u_knots_inner, mode[:, :, spin_ind, 1].T, kx=3, ky=3)

            coeff_arr_out[spin_ind, k ,0] = spl1.tck[2]
            coeff_arr_out[spin_ind, k, 1] = spl2.tck[2]

    regionA.create_dataset("z_knots", data=np.linspace(0,1,33))
    regionA.create_dataset("w_knots", data=spl1.tck[0])
    regionA.create_dataset("u_knots", data=spl1.tck[1])

    # Region B
    regionB = f.create_group('regionB')

    regionB.attrs['NZ'] = 33
    regionB.attrs['NW'] = 17
    regionB.attrs['NU'] = 17

    w_knots_outer = np.linspace(0,1,17)
    u_knots_outer = np.linspace(0,1,17)

    coeff_arr_out_B = regionB.create_dataset("CoeffsRegionB", (33, l_arr_no_mask.size, 2, 17*17), dtype='f8')
    for k, (ell_select, emm_select, enn_select) in tqdm(enumerate(zip(l_arr_no_mask, m_arr_no_mask, n_arr_no_mask))):
        if enn_select >= 0:
            # print(ell_select, emm_select, enn_select)
            mode_path = os.path.join(data_dir_B, f'l{ell_select}', f'm{emm_select}', f'amplitudes_{ell_select}_{emm_select}_{enn_select}.feather')
            modetest = pd.read_feather(mode_path)
            mode = modetest.to_numpy().reshape(33, 33, 33, 2)
        else:
            emm_select = -emm_select
            enn_select = -enn_select

            mode_path = os.path.join(data_dir_B, f'l{ell_select}', f'm{emm_select}', f'amplitudes_{ell_select}_{emm_select}_{enn_select}.feather')
            modetest = pd.read_feather(mode_path)
            mode = modetest.to_numpy().reshape(33, 33, 33, 2)
            mode[:,:,:,1] *= -1
            mode *= (-1) ** ell_select
        
        for spin_ind in range(33):
            spl1 = RectBivariateSpline(w_knots_outer, u_knots_outer, mode[::2, ::2, spin_ind, 0].T, kx=3, ky=3)
            spl2 = RectBivariateSpline(w_knots_outer, u_knots_outer, mode[::2, ::2, spin_ind, 1].T, kx=3, ky=3)

            coeff_arr_out_B[spin_ind, k ,0] = spl1.tck[2]
            coeff_arr_out_B[spin_ind, k, 1] = spl2.tck[2]

    regionB.create_dataset("z_knots", data=np.linspace(0,1,33))
    regionB.create_dataset("w_knots", data=spl1.tck[0])
    regionB.create_dataset("u_knots", data=spl1.tck[1])

    # add datasets for the regionA and regionB parameter grids for completeness
    params_path = os.path.join(data_dir_A, 'params.feather')
    paramsA = pd.read_feather(params_path)
    params = paramsA.to_numpy().reshape(33, 33, 33, 3).transpose(2,0,1,3)
    regionA.create_dataset("ParamsRegionA", data=params)

    params_path = os.path.join(data_dir_B, 'params.feather')
    paramsB = pd.read_feather(params_path)
    params = paramsB.to_numpy().reshape(33, 33, 33, 3)[::2,::2,:, :].transpose(2,0,1,3)
    regionB.create_dataset("ParamsRegionB", data=params)
