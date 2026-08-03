from .base import AmplitudeBase
from ..utils.baseclasses import KerrCirc, BackendLike, xp_ndarray, Union
from typing import Optional, List
import h5py
from ..summation.interpolatedmodesum import CubicSplineInterpolant
import numpy as np


class AmplitudeCirc1PAT1R(AmplitudeBase, KerrCirc):
    """Calculate 1PA Teukolsky amplitudes in the slowly-spinning Kerr circular and arbitrary secondary spin regime with 1D cubic spline interpolation.

    This class interpolates precomputed amplitude data stored in HDF5 files using 1D cubic splines.
    The data is organized by different pieces of the post-adiabatic (PA) expansion: 0PA, 1PA,
    1PAchi1, 1PAchi2, and 1PAdeltam.

    This module is available for GPU and CPU.

    args:
        filename: HDF5 file path containing amplitude data. If None, uses default file.
        force_backend: Backend to use for computation (optional).
        zero_PA_amps_only: If True, only the 0PA amplitudes are computed and all 1PA contributions are set to zero (optional; defaults to False).
        **kwargs: Optional keyword arguments for the base class:
            :class:`few.utils.baseclasses.KerrCirc`.
    """

    num_teuk_modes: int
    """Total number of mode amplitude grids this interpolant stores.
    """

    filename: str

    def __init__(
        self,
        filename: Optional[str] = None,
        force_backend: BackendLike = None,
        zero_PA_amps_only: bool = False,
        **kwargs,
    ):
        AmplitudeBase.__init__(self)
        KerrCirc.__init__(self, force_backend=force_backend, **kwargs)

        self.filename = (
            "Amplitudes1PAT1R.h5" if filename is None else filename
        )

        from few import get_file_manager

        file_path = get_file_manager().get_file(self.filename)

        self.amp_spline = {}
        self.r_min = {}; self.r_max = {}
        self.amp_keys = ["0PA", "1PA", "1PAchi1", "1PAchi2", "1PAdeltam"]
        self.num_teuk_modes = len(self.l_arr_no_mask)

        with h5py.File(file_path, "r") as f:
            # load attributes in the right order for correct mode sorting later
            format_string1 = "/{}/l{}/m{}/Infinity"
            # loop over 0PA, 1PA, 1PAchi1, 1PAchi2
            for data_key in self.amp_keys:
                if data_key != "1PAdeltam":
                    grid = f[f"/{data_key}/Grid"][()]
                    self.r_min[data_key] = grid.min()
                    self.r_max[data_key] = grid.max()
                    num_grid = len(grid)
                    unique_interp = self._map_coord(grid, data_key)
                    mode_data = np.zeros((2 * self.num_teuk_modes, num_grid), dtype=np.float64)

                    for k, (ell, emm) in enumerate(zip(self.l_arr_no_mask, self.m_arr_no_mask)):
                        key1 = format_string1.format(data_key, ell, emm)
                        tmp = f[key1][()]
                        data_here =  tmp[:, 0] + 1j * tmp[:, 1]
                        mode_data[k] = data_here.real
                        mode_data[k + self.num_teuk_modes] = data_here.imag

                    self.amp_spline[data_key] = self.build_with_same_backend(
                        CubicSplineInterpolant,
                        args = [
                            unique_interp, -mode_data#Overall minus sign in waveform to agree with FEW convention
                        ]
                    )
                self.r_min["1PAdeltam"] = self.r_min["0PA"]
                self.r_max["1PAdeltam"] = self.r_max["0PA"];
        self.rmin_max = max(self.r_min.values())
        self.rmax_min = min(self.r_max.values())

        self.zero_PA_amps_only = zero_PA_amps_only

    @classmethod
    def module_references(cls) -> list[REFERENCE]:
        """Return citations related to this module"""
        return [REFERENCE.CIRC_1PAT1R] + super().module_references()

    def _map_coord(self, p, data_key="0PA"):
        return (self.xp.asarray(p) - self.r_min[data_key]) / (self.r_max[data_key] - self.r_min[data_key])

    def _amplitudes_single_piece(self, p, data_key, mode_indices, conj_mode_mask, deriv_order=0
    ) -> Union[dict, np.ndarray]:
        """Generate one of the contributions to the 1PA Teukolsky amplitudes.

        This method evaluates the 1D cubic spline interpolant for a specific piece
        of the post-adiabatic expansion (identified by data_key). The 1PAdeltam piece
        is computed from the derivative of the 0PA piece.

        Args:
            p: Dimensionless semi-latus rectum.
            data_key: Identifier for the PA expansion piece. Options are "0PA", "1PA",
                "1PAchi1", "1PAchi2", "1PAdeltam".
            mode_indices: Indices of modes to be generated.
            conj_mode_mask: Boolean mask for conjugate modes.
            deriv_order: Order of derivative to compute. Default is 0.

        Returns:
            An array of complex mode amplitudes.
        """
        # "1PAdeltam" is not a spline key - compute it from the 0PA derivative
        if data_key == "1PAdeltam":
            dA_dp = self._amplitudes_single_piece(
                p, data_key="0PA", mode_indices=mode_indices, conj_mode_mask=conj_mode_mask, deriv_order=1
            )
            chainrule = 1.0 / (self.r_max["0PA"] - self.r_min["0PA"])
            return -(2./3. * chainrule * self.xp.reshape(p, (-1, 1))) * dA_dp

        if self.xp.any(p > self.r_max[data_key]) or self.xp.any(p < self.r_min[data_key]):
            raise ValueError(f"(AmplitudeCirc1PAT1R) Some values of p lie outside the amplitude data range [{self.r_min[data_key]}, {self.r_max[data_key]}] for the {data_key} piece")

        p_int = self._map_coord(p, data_key=data_key)
        if deriv_order > 0:
            p_int = self.xp.clip(p_int, 0.0, 1.0)

        #  TODO: right now, all modes are ALWAYS computed. Need a better 1-d spline setup.
        teuk_modes = self.xp.atleast_2d(self.amp_spline[data_key](p_int, deriv_order=deriv_order).T)
        teuk_modes = teuk_modes[:, mode_indices] + 1j * teuk_modes[:, mode_indices + self.num_teuk_modes]

        teuk_modes[:, conj_mode_mask] = (
            (-1) ** self.l_arr_no_mask[mode_indices[conj_mode_mask]] * self.xp.conj(teuk_modes[:, conj_mode_mask])
        )
        return teuk_modes

    def get_amplitudes_single_piece(
        self, p, *, data_key="0PA", specific_modes=None, deriv_order=0
    ) -> Union[dict, np.ndarray]:
        """Generate one of the contributions to the 1PA Teukolsky amplitudes.

        This method evaluates the 1D cubic spline interpolant for a specific piece
        of the post-adiabatic expansion (identified by data_key). The 1PAdeltam piece
        is computed from the derivative of the 0PA piece.

        Args:
            p: Dimensionless semi-latus rectum.
            data_key: Identifier for the PA expansion piece. Options are "0PA", "1PA",
                "1PAchi1", "1PAchi2", "1PAdeltam". Default is "0PA".
            specific_modes: Indices of modes to be generated (optional;
                defaults to all modes).
            deriv_order: Order of derivative to compute. Default is 0.

        Returns:
            If specific_modes is a list of tuples, returns a dictionary of complex mode
            amplitudes with mode tuples as keys. Otherwise, returns an array of complex
            mode amplitudes.
        """
        if data_key not in self.amp_keys:
            raise ValueError(f"(AmplitudeCirc1PAT1R) key \'{data_key}\' is not in accepted list of amplitude pieces {self.amp_keys}")

        if self.xp.any(p > self.r_max[data_key]) or self.xp.any(p < self.r_min[data_key]):
            raise ValueError(f"(AmplitudeCirc1PAT1R) Some values of p lie outside the amplitude data range [{self.r_min[data_key]}, {self.r_max[data_key]}] for the {data_key} piece")

        # select modes
        if specific_modes is None:
            mode_indices = self.xp.arange(self.num_teuk_modes)
            conj_mode_mask = self.xp.zeros_like(mode_indices, dtype=bool)
        else:
            assert isinstance(specific_modes, (self.xp.ndarray)), (
                f"(AmplitudeCirc1PAT1R) specific_modes must be a one dimensional array of mode indicies"
            )
            mode_indices = specific_modes.copy()
            conj_mode_mask = mode_indices >= self.num_teuk_modes
            mode_indices[conj_mode_mask] -= self.num_m_1_up

        return self._amplitudes_single_piece(p, data_key, mode_indices, conj_mode_mask, deriv_order=deriv_order)

    def _combine_pieces(self, data_pieces, nu, chit1, chit2, deltaM, mode_indices):
        #TODO: add bounds tests on chit1, chit2, deltaM (if needed)
        overall_factor = self.xp.sqrt(1 - 4 * nu)

        #Factored out of mass-ratio expansion for odd modes
        chit1_r     = self.xp.reshape(chit1,     (-1, 1))
        deltaM_r = self.xp.reshape(deltaM, (-1, 1))

        combined = (
            data_pieces["0PA"] + nu * data_pieces["1PA"]
            + chit1_r     * data_pieces["1PAchi1"]
            + chit2      * data_pieces["1PAchi2"]
            + nu * deltaM_r * data_pieces["1PAdeltam"]
        )

        # Per-mode mask for odd m: shape (1, num_teuk_modes)
        odd_m_mask = self.xp.asarray(self.m_arr_no_mask % 2 != 0)[None, mode_indices]
        odd_m_correction = self.xp.where(odd_m_mask, 2 * nu * data_pieces["0PA"], 0.0)
        scale = self.xp.where(odd_m_mask, overall_factor, 1.0)
        return scale * (combined + odd_m_correction)

    def get_amplitudes(
        self, a, p, e, xI, nu, chit1, chit2, deltaM, specific_modes=None
    ):
        """Generate the 1PA Teukolsky amplitudes.

        Args:
            a: Primary spin
            p: Dimensionless semi-latus rectum.
            e: Eccentricity (must be zero).
            xI: Cosine of orbital inclination (must be 1.0).
            nu: Symmetric mass ratio (optional; defaults to 0.0).
            chit2: Secondary reduced dimensionless spin (optional; defaults to 0.0).
            deltaM: Deviation from initial total mass (optional; defaults to 0.0).
            chit1: Primary reduced dimensionless spin (optional; defaults to 0.0).
            specific_modes: Indices of modes to be generated (optional; defaults to all modes).
        """

        assert self.xp.all(e == 0.0), "(AmplitudeCirc1PAT1R) Quasicircular model: e must be identically 0"
        assert self.xp.all(xI == 1.0), "(AmplitudeCirc1PAT1R) Equatorial model: xI must be identically 1"

        if self.xp.any(self.xp.asarray(nu) < 0.0) or self.xp.any(self.xp.asarray(nu) > 0.25):
            raise ValueError(f"(AmplitudeCirc1PAT1R) nu={nu} is outside the physical range [0, 0.25].")

        # select modes. TODO: right now, all modes are ALWAYS computed. Need a better 1-d spline setup.
        if specific_modes is None:
            mode_indices = self.xp.arange(self.num_teuk_modes)
            conj_mode_mask = self.xp.zeros_like(mode_indices, dtype=bool)
        else:
            assert isinstance(specific_modes, (self.xp.ndarray)), (
                f"(AmplitudeCirc1PAT1R) specific_modes must be a one dimensional array of mode indices"
            )
            mode_indices = specific_modes.copy()
            conj_mode_mask = mode_indices >= self.num_teuk_modes
            mode_indices[conj_mode_mask] -= self.num_m_1_up

        data_pieces = {}

        # Always compute the 0PA piece
        data_key = "0PA"
        data_pieces[data_key] = self._amplitudes_single_piece(
            p, data_key, mode_indices, conj_mode_mask
        )

        if self.zero_PA_amps_only:
            # Fill all 1PA pieces with zeros of the same shape/type as 0PA
            zero = self.xp.zeros_like(data_pieces["0PA"])
            for key in ["1PA", "1PAchi1", "1PAchi2", "1PAdeltam"]:
                data_pieces[key] = zero
        else:
            for data_key in ["1PA", "1PAchi1", "1PAchi2", "1PAdeltam"]:
                data_pieces[data_key] = self._amplitudes_single_piece(
                    p, data_key, mode_indices, conj_mode_mask
                )

        return self._combine_pieces(data_pieces, nu, chit1, chit2, deltaM, mode_indices)

    def __reduce__(self):
        return (self.__class__, (self.filename,))