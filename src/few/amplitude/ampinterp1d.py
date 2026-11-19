from .base import AmplitudeBase
from ..utils.baseclasses import KerrCirc, BackendLike, xp_ndarray, Union
from ..utils.citations import REFERENCE
from ..utils.mappings.common import chi_to_chit
from typing import Optional, List
import h5py
from ..summation.interpolatedmodesum import CubicSplineInterpolant
import numpy as np


class AmplitudeCirc1PAT1R(AmplitudeBase, KerrCirc):
    """Calculate 1PA Teukolsky amplitudes in the slowly-spinning Kerr circular and arbitrary secondary spin regime with 1D cubic spline interpolation.

    This class interpolates precomputed amplitude data stored in HDF5 files using 1D cubic splines.
    The data is organized into different terms in the post-adiabatic (PA) expansion: 0PA, 1PA,
    1PAchi1, 1PAchi2, and 1PAdeltam, corresponding to the amplitudes in Eq. (138) of 2510.16113.
    Odd azimuthal modes are additionally resummed as described in 2112.12265 (see text description of Fig. 2 therein).

    This module is available for GPU and CPU.

    args:
        filename: HDF5 file path containing amplitude data. If None, uses default file.
        force_backend: Backend to use for computation (optional).
        zero_PA_amps_only: If True, all 1PA contributions are set to zero. Note 0PA odd azimuthal modes are still resummed. (Optional; defaults to False.)
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
            "Amplitudes1PAT1R_v2.h5" if filename is None else filename
        )

        from few import get_file_manager

        file_path = get_file_manager().get_file(self.filename)

        self.amp_spline = {}
        self.r_min = {}; self.r_max = {}
        self.amp_keys = ["0PA", "1PA", "1PAchi1", "1PAchi2", "1PAdeltam"]
        self.num_teuk_modes = len(self.l_arr_no_mask)
        self._piece_power = {}  # the power n of each mode of each spline (see below)
        self._m_modes = self.xp.asarray(self.m_arr_no_mask, dtype=self.xp.float64)

        with h5py.File(file_path, "r") as f:
            # load attributes in the right order for correct mode sorting later
            # each piece is stored as |A/x^n| and arg(A/x^n) - 2 w ln w (w = m v^3, v = sqrt(x), n the attribute "n" of
            # each dataset) on an increasing grid in x = 1/p, which may include x = 0, and splined in v; m = 0 modes
            # vanish and are not stored. 1PA is stored as two pieces, d2G and E1h1 (see _amplitudes_single_piece).
            format_string1 = "/{}/l{}/m{}/{}"
            for data_key in ["0PA", "1PA/d2G", "1PA/E1h1", "1PAchi1", "1PAchi2"]:
                group, _, piece = data_key.partition("/")
                grid = f[f"/{group}/Grid"][()]
                self.r_min[data_key] = self.r_min[group] = 1.0 / grid.max()
                self.r_max[data_key] = self.r_max[group] = np.inf if grid.min() == 0 else 1.0 / grid.min()
                num_grid = len(grid)
                unique_interp = self.xp.asarray(np.sqrt(grid))
                mode_data = np.zeros((2 * self.num_teuk_modes, num_grid), dtype=np.float64)
                power = np.zeros(self.num_teuk_modes)

                for k, (ell, emm) in enumerate(zip(self.l_arr_no_mask, self.m_arr_no_mask)):
                    if emm == 0:
                        continue
                    ds = f[format_string1.format(group, ell, emm, piece or "Infinity")]
                    tmp = ds[()]
                    mode_data[k] = tmp[:, 0]
                    mode_data[k + self.num_teuk_modes] = tmp[:, 1]
                    power[k] = ds.attrs["n"]

                self._piece_power[data_key] = self.xp.asarray(power)
                self.amp_spline[data_key] = self.build_with_same_backend(
                    CubicSplineInterpolant,
                    args = [
                        unique_interp, mode_data
                    ]
                )
            self.r_min["1PAdeltam"] = self.r_min["0PA"]
            self.r_max["1PAdeltam"] = self.r_max["0PA"]

        self.zero_PA_amps_only = zero_PA_amps_only

    @classmethod
    def module_references(cls) -> list[REFERENCE]:
        """Return citations related to this module"""
        return [REFERENCE.CIRC_1PAT1R] + super().module_references()

    def _map_coord(self, p):
        return 1.0 / self.xp.sqrt(self.xp.asarray(p, dtype=self.xp.float64))

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
        # "1PAdeltam" is not a spline key. 
        # Instead the coefficient of deltaM in the 1PA amplitudes is computed from the 0PA amplitudes
        # using Eq. (120) of 2510.16113 and applying the same reasoning as below Eq. (89). 
        # Corrected by Eq. (5.1) of the technical document. 
        if data_key == "1PAdeltam":
            A = self._amplitudes_single_piece(
                p, data_key="0PA", mode_indices=mode_indices, conj_mode_mask=conj_mode_mask
            )
            dA_dp = self._amplitudes_single_piece(
                p, data_key="0PA", mode_indices=mode_indices, conj_mode_mask=conj_mode_mask, deriv_order=1
            )
            return -(2./3. * self.xp.reshape(p, (-1, 1))) * dA_dp - A

        if self.xp.any(p > self.r_max[data_key]) or self.xp.any(p < self.r_min[data_key]):
            raise ValueError(f"(AmplitudeCirc1PAT1R) Some values of p lie outside the amplitude data range [{self.r_min[data_key]}, {self.r_max[data_key]}] for the {data_key} piece")

        # The 1PA piece, the first line of Eq. (2.13) of the technical document (the extra 2 A_0PA of the
        # odd-m modes is added when the pieces are combined):
        #   A_1PA + A_0PA + (2/3) (1 + dM) p dA_0PA/dp,   dM = -(1/p) / sqrt(1 - 3/p),
        # with A_1PA = d2G + E1h1 / (p - 6) and d2G and E1h1 splined; E1h1 is the term with the 0PA r0dot
        # (which diverges at p = 6) times p - 6.
        if data_key == "1PA":
            if deriv_order != 0:
                raise ValueError("(AmplitudeCirc1PAT1R) only deriv_order 0 is available for the 1PA piece")
            pc = self.xp.reshape(self.xp.asarray(p, dtype=self.xp.float64), (-1, 1))
            dM = -(1.0 / pc) / self.xp.sqrt(1.0 - 3.0 / pc)
            piece = lambda key, deriv_order=0: self._amplitudes_single_piece(
                p, key, mode_indices, conj_mode_mask, deriv_order=deriv_order)
            return (piece("1PA/d2G") + piece("1PA/E1h1") / (pc - 6.0) + piece("0PA")
                    + (2.0 / 3.0) * (1.0 + dM) * pc * piece("0PA", 1))

        if deriv_order > 1:
            raise ValueError(f"(AmplitudeCirc1PAT1R) only deriv_order 0 and 1 are available for the {data_key} piece")

        # A = |A/x^n| v^(2n) exp(i (arg(A/x^n) - 2 w ln w + 2 w ln w)), from the splines in v = sqrt(x) = 1/sqrt(p)
        p_int = self._map_coord(p)
        v = self.xp.reshape(p_int, (-1, 1))
        n = self._piece_power[data_key][mode_indices]
        m = self._m_modes[mode_indices]
        w = m * v**3
        log_w = self.xp.log(self.xp.where(w > 0, w, 1.0))  # w ln w and v^2 ln w vanish at w = 0

        #  TODO: right now, all modes are ALWAYS computed. Need a better 1-d spline setup.
        spl = self.xp.atleast_2d(self.amp_spline[data_key](p_int).T)
        abs_A, phi = spl[:, mode_indices], spl[:, mode_indices + self.num_teuk_modes]
        teuk_modes = -v ** (2 * n) * self.xp.exp(1j * (phi + 2 * w * log_w))  # Overall minus sign in waveform to agree with FEW convention
        if deriv_order == 0:
            teuk_modes = abs_A * teuk_modes
        else:  # dA/dp = -(v^3/2) dA/dv
            dspl = self.xp.atleast_2d(self.amp_spline[data_key](p_int, deriv_order=1).T)
            dabs_A, dphi = dspl[:, mode_indices], dspl[:, mode_indices + self.num_teuk_modes]
            dpsi = dphi + 6 * m * v**2 * (log_w + 1)  # d/dv of the full phase arg(A/x^n)
            teuk_modes = v**2 * (-0.5 * teuk_modes * (2 * n * abs_A + v * (dabs_A + 1j * abs_A * dpsi)))

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
        is computed from the 0PA piece and its first derivative. Note the 0PA piece will
        differ from the complete waveform amplitude even if zero_PA_amps_only is True,
        since the odd azimuthal modes are resummed in the complete amplitude.

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
        self, a, p, e, xI, nu, chi2, deltachit1, deltaM, specific_modes=None
    ):
        """Generate the 1PA Teukolsky amplitudes.

        Args:
            a: Initial dimensionless primary spin [i.e. chi1(t=0)]
            p: Dimensionless semi-latus rectum.
            e: Eccentricity (must be zero).
            xI: Cosine of orbital inclination (must be 1.0).
            nu: Symmetric mass ratio.
            chi2: Secondary dimensionless spin.
            deltachit1: Deviation from the initial primary reduced spin.
            deltaM: Deviation from initial total mass.
            specific_modes: Indices of modes to be generated (optional; defaults to all modes).
        """

        assert self.xp.all(e == 0.0), "(AmplitudeCirc1PAT1R) Quasicircular model: e must be identically 0"
        assert self.xp.all(xI == 1.0), "(AmplitudeCirc1PAT1R) Equatorial model: xI must be identically 1"

        if self.xp.any(self.xp.asarray(nu) < 0.0) or self.xp.any(self.xp.asarray(nu) > 0.25):
            raise ValueError(f"(AmplitudeCirc1PAT1R) nu={nu} is outside the physical range [0, 0.25].")

        # Convert (nu, chi2, deltachit1) -> (chit1(t), chit2)
        chit1_0, chit2 = chi_to_chit(a, chi2, nu)
        chit1 = chit1_0 + deltachit1

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
            # Note 0PA odd azimuthal modes are still resummed
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