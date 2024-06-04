# Flux-based Schwarzschild Eccentric amplitude module for Fast EMRI Waveforms
# performs calculation with a Roman network

# Copyright (C) 2020 Michael L. Katz, Alvin J.K. Chua, Niels Warburton, Scott A. Hughes
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from copy import deepcopy
import os
import warnings

import numpy as np
import h5py
from scipy.interpolate import RectBivariateSpline
from tqdm import tqdm

# Cython/C++ imports

# Python imports
from few.utils.baseclasses import SchwarzschildEccentric, AmplitudeBase, ParallelModuleBase, KerrEquatorialEccentric
from few.utils.utility import check_for_file_download
from few.utils.citations import *
from few.utils.utility import p_to_y, kerr_p_to_u

# check for cupy and GPU version of pymatmul
try:
    # Cython/C++ imports
    from pyAmpInterp2D import interp2D
    # Python imports
    import cupy as cp

except (ImportError, ModuleNotFoundError) as e:
    import numpy as np

from pyAmpInterp2D_cpu import interp2D as interp2D_cpu


# get path to this file
dir_path = os.path.dirname(os.path.realpath(__file__))


class AmpInterp2D(AmplitudeBase, KerrEquatorialEccentric, ParallelModuleBase):
    """Calculate Teukolsky amplitudes with a ROMAN.

    ROMAN stands for reduced-order models with artificial neurons. Please see
    the documentations for
    :class:`few.utils.baseclasses.SchwarzschildEccentric`
    for overall aspects of these models.

    A reduced order model is computed for :math:`A_{lmn}`. The data sets that
    are provided over a grid of :math:`(p,e)` were provided by Scott Hughes.

    A feed-foward neural network is then trained on the ROM. Its weights are
    used in this module.

    When the user inputs :math:`(p,e)`, the neural network determines
    coefficients for the modes in the reduced basic and transforms it back to
    amplitude space.

    This module is available for GPU and CPU.


    args:
        max_init_len (int, optional): Number of points to initialize for
            buffers. This allows the user to limit memory usage. However, if the
            user requests more length, a warning will be thrown and the
            max_init_len will be increased accordingly and arrays reallocated.
            Default is 1000.
        **kwargs (dict, optional): Keyword arguments for the base classes:
            :class:`few.utils.baseclasses.SchwarzschildEccentric`,
            :class:`few.utils.baseclasses.AmplitudeBase`,
            :class:`few.utils.baseclasses.ParallelModuleBase`.
            Default is {}.

    """

    def attributes_AmpInterp2D(self):
        """
        attributes:
            few_dir (str): absolute path to the FastEMRIWaveforms directory
            break_index (int): length of output vector from network divded by 2.
                It is really the number of pairs of real and imaginary numbers.
            use_gpu (bool): If True, use the GPU.
            neural_layer (obj): C++ class for computing neural network operations
            transform_output(obj): C++ class for transforming output from
                neural network in the reduced basis to the full amplitude basis.
            num_teuk_modes (int): number of teukolsky modes in the data file.
            transform_factor_inv (double): Inverse of the scalar transform factor.
                For this model, that is 1000.0.
            max_init_len (int): This class uses buffers. This is the maximum length
                the user expects for the input arrays.
            weights (list of xp.ndarrays): List of the weight matrices for each
                layer of the neural network. They are flattened for entry into
                C++ in column-major order. They have shape (dim1, dim2).
            bias (list of xp.ndarrays): List of the bias arrays for each layer
                of the neural network. They have shape (dim2,).
            dim1 (list of int): List of 1st dimension length in each layer.
            dim2 (list of int): List of 2nd dimension length in each layer.
            num_layers (int): Number of layers in the neural network.
            transform_matrix (2D complex128 xp.ndarray): Matrix for tranforming
                output of neural network onto original amplitude basis.
            max_num (int): Figures out the maximum dimension of all weight matrices
                for buffers.
            temp_mats (len-2 list of double xp.ndarrays): List that holds
                temporary matrices for neural network evaluation. Each layer switches
                between which is the input and output to properly interface with
                cBLAS/cuBLAS.
            run_relu_arr  (1D int xp.ndarray): Array holding information about
                whether each layer will run the relu activation. All layers have
                value 1, except for the last layer with value 0.

        """
        pass

    def __init__(self, fp, max_init_len=1000, **kwargs):

        ParallelModuleBase.__init__(self, **kwargs)
        KerrEquatorialEccentric.__init__(self, **kwargs)
        AmplitudeBase.__init__(self, **kwargs)

        if self.use_gpu:
            self.interp2D = interp2D
            xp = cp
        else:
            self.interp2D = interp2D_cpu
            xp = np

        self.few_dir = dir_path + "/../../"

        # check if user has the necessary data
        # if not, the data will automatically download

        coefficients = h5py.File(self.few_dir+"few/files/" + fp

        self.a_val_store = coefficients.attrs['signed_spin']

        self.num_teuk_modes = coefficients.attrs['num_teuk_modes']
        self.tck = [
            xp.asarray(coefficients['x1']), 
            xp.asarray(coefficients['x2']), 
            xp.asarray(coefficients['c'])
        ]
        self.degrees = coefficients.attrs['spline_degree_x'], coefficients.attrs['spline_degree_y']
        self.len_indiv_c = coefficients.attrs['points_per_modegrid']

    @property
    def citation(self):
        """Return citations for this module"""
        return romannet_citation + larger_few_citation + few_citation + few_software_citation

    @property
    def gpu_capability(self):
        """Confirms GPU capability"""
        return True

    def __call__(self, a, p, e, xI, *args, specific_modes=None, **kwargs):
        """
        Evaluate the spline or its derivatives at given positions.
        Parameters
        ----------
        x, y : array_like
            Input coordinates.
            If `grid` is False, evaluate the spline at points ``(x[i],
            y[i]), i=0, ..., len(x)-1``.  Standard Numpy broadcasting
            is obeyed.
            If `grid` is True: evaluate spline at the grid points
            defined by the coordinate arrays x, y. The arrays must be
            sorted to increasing order.
            Note that the axis ordering is inverted relative to
            the output of meshgrid.
        
        """
        
        if self.use_gpu:
            xp = cp
        else:
            xp = np

        grid = False

        try:
            a_cpu, p_cpu, e_cpu, xI_cpu = a.get().copy(), p.get().copy(), e.get().copy(), xI.get().copy()
        except AttributeError:
            a_cpu, p_cpu, e_cpu, xI_cpu = a.copy(), p.copy(), e.copy(), xI.copy()

        a = xp.asarray(a)
        p = xp.asarray(p)
        e = xp.asarray(e)
        xI = xp.asarray(xI)

        assert xp.all(a == self.a_val_store)
        a_cpu *= xI_cpu  # correct the sign of a now we've passed the check, for the reparameterisation
        # TODO: make this GPU accessible
        u = xp.asarray(kerr_p_to_u(a_cpu, p_cpu, e_cpu, xI_cpu, use_gpu=False))

        w = xp.sqrt(e)

        tw, tu, c = self.tck[:3]
        kw, ku = self.degrees
        
        # standard Numpy broadcasting
        if w.shape != u.shape:
            w, u = np.broadcast_arrays(w, u)

        shape = w.shape
        w = w.ravel()
        u = u.ravel()

        if w.size == 0 or u.size == 0:
            return np.zeros(shape, dtype=self.tck[2].dtype)

        nw = tw.shape[0]
        nu = tu.shape[0]
        mw = w.shape[0]
        mu = u.shape[0]

        assert mw == mu

        # TODO: adjustable

        if specific_modes is None:
            mode_indexes = xp.arange(self.num_teuk_modes)
        
        else:
            if isinstance(specific_modes, xp.ndarray):
                mode_indexes = specific_modes
            elif isinstance(specific_modes, list):  # the following is slow and kills efficiency
                mode_indexes = xp.zeros(len(specific_modes), dtype=xp.int32)
                for i, (l, m, n) in enumerate(specific_modes):
                    try:
                        mode_indexes[i] = np.where((self.l_arr == l) & (self.m_arr == m) & (self.n_arr == n))[0]
                    except:
                        raise Exception(f"Could not find mode index ({l},{m},{n}).")
        # TODO: perform this in the kernel
        c_in = c[mode_indexes].flatten()

        num_indiv_c = 2*len(mode_indexes)  # Re and Im
        len_indiv_c = self.len_indiv_c

        z = xp.zeros((num_indiv_c * mw))
        
        self.interp2D(z, tw, nw, tu, nu, c_in, kw, ku, w, mw, u, mu, num_indiv_c, len_indiv_c)

        #check = np.asarray([[spl.ev(e.get(), y.get()) for spl in spl1] for spl1 in self.spl2D.values()]).transpose(2, 1, 0)

        z = z.reshape(num_indiv_c//2, 2, mw).transpose(2, 1, 0)

        z = z[:, 0] + 1j * z[:, 1]
        return z


class AmpInterpKerrEqEcc(AmplitudeBase, KerrEquatorialEccentric, ParallelModuleBase):
    def __init__(self, specific_spins=None, **kwargs):

        ParallelModuleBase.__init__(self, **kwargs)
        KerrEquatorialEccentric.__init__(self, **kwargs)
        AmplitudeBase.__init__(self, **kwargs)

        self.few_dir = dir_path + "/../../"


    # outfile = h5py.File(few_dir+"few/files/" + f"KerrEqEccAmpCoeffs_a{abs(a_val_store):.3f}{part2}.h5","w")

        if specific_spins is None:
            spins_tmp = []
            for fp in os.listdir(self.few_dir + "few/files/"):
                if fp[:20] == "KerrEqEccAmpCoeffs_a":
                    spin_h = float(fp[20:24])
                    if fp[24:27] == "_r_":
                        spin_h *= -1  # retrograde
                    spins_tmp.append(spin_h)
            # combine prograde and retrograde here
            self.spin_values = np.unique(np.asarray(spins_tmp))
        else:
            self.spin_values = np.asarray(specific_spins)
        

        self.spin_information_holder = [None for _ in self.spin_values]
        for i, spin in enumerate(self.spin_values):
            base_string = f"{abs(spin):1.3f}"
            if spin < 0.0:
                base_string += "_r"
            elif spin > 0.0:
                base_string += "_p"
            fp = f"KerrEqEccAmpCoeffs_a{base_string}.h5"  # coefficient files computed

            self.spin_information_holder[i] = AmpInterp2D(fp, use_gpu=self.use_gpu)

        if self.use_gpu:
            xp = cp
        else:
            xp = np
        
        pos_neg_n_swap_inds = []
        if self.use_gpu:
            for l,m,n in zip(self.l_arr_no_mask.get(),self.m_arr_no_mask.get(),self.n_arr_no_mask.get()):
                pos_neg_n_swap_inds.append(self.special_index_map[(l,m,-n)])
        else:
            for l,m,n in zip(self.l_arr_no_mask,self.m_arr_no_mask,self.n_arr_no_mask):
                pos_neg_n_swap_inds.append(self.special_index_map[(l,m,-n)])
            
        self.pos_neg_n_swap_inds = xp.asarray(pos_neg_n_swap_inds)

    def get_amplitudes(self, a, p, e, xI, specific_modes=None):
        if self.use_gpu:
            xp = cp
        else:
            xp = np

        # prograde: spin pos, xI pos
        # retrograde: spin pos, xI neg - >  spin neg, xI pos
        assert isinstance(a, float)

        assert np.all(xI == 1.0) or np.all(xI == -1.0)  # either all prograde or all retrograde
        xI_in = np.ones_like(p)*xI
        
        signed_spin = a * xI_in[0].item()

        if signed_spin in self.spin_values:
            ind_1 = np.where(self.spin_values == signed_spin)[0][0]
            a_in = np.full_like(p, signed_spin)

            z = self.spin_information_holder[ind_1](a_in, p, e, xI_in, specific_modes=specific_modes)
            if xI_in[0] == -1 and signed_spin != 0.:  # retrograde needs mode flip
                z = xp.conj(z[:,self.pos_neg_n_swap_inds])

        else:
            ind_above = np.where(self.spin_values > signed_spin)[0][0]
            ind_below = ind_above - 1
            assert ind_above < len(self.spin_values)
            assert ind_below >= 0

            a_above = np.full_like(p, self.spin_values[ind_above])
            a_above_single = a_above[0]
            assert np.all(a_above_single == a_above[0])

            a_below = np.full_like(p, self.spin_values[ind_below])
            a_below_single = a_below[0]
            assert np.all(a_below_single == a_below[0])

            # handle retrograde mode flip (n -> conj(-n))

            if a_below_single < 0:
                apply_conjugate_below = True
                if specific_modes is None:
                    specific_modes_below = self.pos_neg_n_swap_inds
                elif isinstance(specific_modes, xp.ndarray):
                    specific_modes_below = self.pos_neg_n_swap_inds[specific_modes]
                elif isinstance(specific_modes, list):
                    specific_modes_below = []
                    for (l, m, n) in specific_modes:
                        specific_modes_below.append((l,m,-n))
            else:
                apply_conjugate_below = False
                specific_modes_below = specific_modes

            if a_above_single < 0:
                apply_conjugate_above = True
                specific_modes_above = specific_modes_below
            else:
                apply_conjugate_above = False
                specific_modes_above = specific_modes
            
            if apply_conjugate_above and apply_conjugate_below:  # combine the flags to save a conj call if both retrograde
                apply_conjugate_total = True
                apply_conjugate_above = False
                apply_conjugate_below = False
            else:
                apply_conjugate_total = False

            z_above = self.spin_information_holder[ind_above](a_above, p, e, xI_in, specific_modes=specific_modes_above)
            z_below = self.spin_information_holder[ind_below](a_below, p, e, xI_in, specific_modes=specific_modes_below)
            if apply_conjugate_below:
                z_below = z_below.conj()
            if apply_conjugate_above:
                z_above = z_above.conj()
            z = ((z_above - z_below) / (a_above_single - a_below_single)) * (signed_spin - a_below_single) + z_below
            if apply_conjugate_total:
                z = z.conj()
        if not isinstance(specific_modes, list):
            return z
        
        # dict containing requested modes
        else:
            temp = {}
            for i, lmn in enumerate(specific_modes):
                temp[lmn] = z[:, i]
                l, m, n = lmn

                # apply +/- m symmetry
                if m < 0:
                    temp[lmn] = np.conj(temp[lmn])

            return temp

def _spline_coefficients_to_file(fp, l_arr, m_arr, n_arr):
    few_dir = dir_path + "/../../"

    # check if user has the necessary data
    # if not, the data will automatically download
    check_for_file_download(fp, few_dir)

    data = {}
    # get information about this specific model from the file
    with h5py.File(few_dir + "few/files/" + fp, "r") as f:
        # load attributes in the right order for correct mode sorting later
        kerr_format_string = "l{}m{}k0n{}"
        grid = f["grid"][:]
        for l, m, n in zip(l_arr, m_arr, n_arr):
            if m >= 0:
                key1 = kerr_format_string.format(l,m,n)
                tmp = f[key1][:]
                tmp2 = tmp[:, 0] + 1j * tmp[:, 1]
                data[key1] = tmp2

    # create the coefficients file

    # adjust the grid
    a = grid.T[0].copy()
    p = grid.T[1].copy()
    e = grid.T[2].copy()
    xI = grid.T[3].copy()
    u = np.round(grid.T[4].copy(),8)  # fix rounding errors in the files
    sep = grid.T[5].copy()
    w = grid.T[6].copy()

    assert np.all(a == a[0])
    assert np.all(xI == xI[0])

    # retrograde needs sign flip to be applied to a
    a *= xI
    a_val_store = a[0]

    if a_val_store < 0:
        part2 = '_r'
    elif a_val_store > 0:
        part2 = '_p'
    elif a_val_store == 0:
        part2 = ''

    outfile = h5py.File(few_dir+"few/files/" + f"KerrEqEccAmpCoeffs_a{abs(a_val_store):.3f}{part2}.h5","w")
    outfile.attrs['signed_spin'] = a_val_store

    grid_size = p.shape[0]
    
    unique_u = np.unique(u)
    unique_w = np.unique(w)
    num_u = len(unique_u)
    num_w = len(unique_w)

    data_copy = deepcopy(data)
    for mode, vals in data.items():
        data_copy[mode] = data_copy[mode].reshape(num_w, num_u)

    data = deepcopy(data_copy)

    data = {name: val[:, ::-1] for name, val in data.items()}

    spl2D = {name:
        [
            RectBivariateSpline(unique_w, unique_u, val.real, kx=3, ky=3), 
            RectBivariateSpline(unique_w, unique_u, val.imag, kx=3, ky=3)
        ]
    for name, val in data.items()}

    mode_keys = list(data.keys())
    num_teuk_modes = len(mode_keys)

    outfile.attrs['num_teuk_modes'] = num_teuk_modes

    first_key = list(spl2D.keys())[0]
    example_spl = spl2D[first_key][0]
    tck_last_entry = np.zeros((len(data), 2, grid_size))
    for i, mode in enumerate(mode_keys):
        tck_last_entry[i, 0] = spl2D[mode][0].tck[2]
        tck_last_entry[i, 1] = spl2D[mode][1].tck[2]

    degrees = example_spl.degrees

    len_indiv_c = tck_last_entry.shape[-1]

    outfile.attrs['spline_degree_x'] = degrees[0]
    outfile.attrs['spline_degree_y'] = degrees[1]
    outfile.attrs['points_per_modegrid'] = len_indiv_c

    outfile.create_dataset('x1', data=example_spl.tck[0])
    outfile.create_dataset('x2', data=example_spl.tck[1])
    outfile.create_dataset('c', data=tck_last_entry.copy())

    outfile.close()

if __name__ == "__main__":
    # produce spline coefficient files (will overwrite previous files if they exist!)
    baseclass = KerrEquatorialEccentric()
    few_dir = dir_path + "/../../"

    spin_values = np.r_[np.linspace(0.,0.9,10),0.95,0.99]
    spin_values = np.r_[-np.flip(spin_values)[:-1],spin_values]
    # or alternatively, specify your own values

    # get a list of amplitude files to compute coefficients for
    base_path = "Teuk_amps_a{:.2f}_{}lmax_10_nmax_50_new_m+.h5"
    filepaths = []
    for spin in spin_values:
        part1 = abs(spin)
        if spin < 0:
            part2 = 'r_'
        elif spin > 0:
            part2 = 'p_'
        elif spin == 0:
            part2 = ''
        filepaths.append(base_path.format(part1, part2))

    # for fp in tqdm(filepaths,total=len(filepaths)):
    #     _spline_coefficients_to_file(fp, baseclass.l_arr, baseclass.m_arr, baseclass.n_arr)

    # test
    amp = AmpInterpKerrEqEcc()
    print(amp.get_amplitudes(0.1, np.asarray([10.]), np.asarray([0.3]), np.asarray([1.])))