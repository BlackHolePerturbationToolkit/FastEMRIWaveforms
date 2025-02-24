import numpy as np
import pandas as pd
import os
import h5py

#  set to the location of the raw data files
data_loc = "./raw_data"

with h5py.File('KerrEccEqFluxData.h5', 'w') as outfile:
    # region A
    dataA = pd.read_feather(os.path.join(data_loc,"fluxes_regionA_raw_65_65_65_fixed.feather"))
    df_infoA = pd.read_feather(os.path.join(data_loc, 'flux_grid_params_inner.feather'))

    NU = np.int64(df_infoA['nu'][0])
    NW = np.int64(df_infoA['nw'][0])
    NZ = np.int64(df_infoA['nz'][0])

    regionA = outfile.create_group('regionA')
    regionA.create_dataset('Edot', data=dataA["Edot"].to_numpy().reshape(NU, NW, NZ)[:,:,:])
    regionA.create_dataset('Ldot', data=dataA["Ldot"].to_numpy().reshape(NU, NW, NZ)[:,:,:])

    regionA.attrs['NU'] = NU
    regionA.attrs['NW'] = NW
    regionA.attrs['NZ'] = NZ

    # region B
    dataB = pd.read_feather(os.path.join(data_loc,"fluxes_regionB_65_33_33.feather"))
    df_infoB = pd.read_feather(os.path.join(data_loc, 'flux_grid_params_outer.feather'))

    NU = np.int64(df_infoB['nu'][0])
    NW = np.int64(df_infoB['nw'][0])
    NZ = np.int64(df_infoB['nz'][0])

    regionB = outfile.create_group('regionB')
    regionB.create_dataset('Edot', data=dataB["Edot"].to_numpy().reshape(NU, NW, NZ)[:,:,:])
    regionB.create_dataset('Ldot', data=dataB["Ldot"].to_numpy().reshape(NU, NW, NZ)[:,:,:])

    regionB.attrs['NU'] = NU
    regionB.attrs['NW'] = NW
    regionB.attrs['NZ'] = NZ