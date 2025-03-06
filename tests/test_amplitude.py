# To run this, open terminal into tests folder and run:
# python -m unittest file_name.Class_Name.Function_to_be_tested
import unittest
import numpy as np
from few.amplitude.ampinterp2d import AmpInterpKerrEqEcc

from few.utils.globals import get_logger

few_logger = get_logger()

few_logger.warning("Amplitude Test is running")

np.random.seed(42)

class AmplitudeTest(unittest.TestCase):
    def test_amplitude_symmetry_relation(self):
        few_logger.info("Testing amplitudes symmetry relation")
        # initialize amplitude class
        kerr_amp_spline = AmpInterpKerrEqEcc()

        # set initial parameters
        a = 0.87
        p = 7.2
        e = 0.4
        xI = 1.0

        specific_modes = [(4,2,1),(5,3,2)]
        specific_modes_symmetry = [(4,-2,-1),(5,-3,-2)]
        sphericalHarmonicFactors = [1,-1]
        amp = kerr_amp_spline(a, p, e, xI, specific_modes=specific_modes)
        amp_symmetry = kerr_amp_spline(a, p, e, xI, specific_modes=specific_modes_symmetry)

        # We should have A(l,-m,-n) = (-1)^l conj(A(l,m,n))
        for i in range(0,len(specific_modes)):
            self.assertAlmostEqual(
                amp[specific_modes[i]], 
                sphericalHarmonicFactors[i] * np.conj(amp_symmetry[specific_modes_symmetry[i]]),
                places=8
                )

    def test_special_index_map(self):
        few_logger.info("Testing special_index_map")
        # initialize amplitude class
        kerr_amp_spline = AmpInterpKerrEqEcc()

        # set initial parameters
        a = 0.87
        p = 7.2
        e = 0.4
        xI = 1.0

        specific_modes = [(2,2,1),(2,-2,1),(2,2,-1),(2,-2,-1),(5,4,2),(5,4,-2),(5,-4,-2),(5,-4,2)]
        all_amplitudes = kerr_amp_spline(a, p, e, xI)
        specific_amplitudes = kerr_amp_spline(a, p, e, xI, specific_modes=specific_modes)
        mode_locations = np.array([kerr_amp_spline.special_index_map[lmn] for lmn in specific_modes])

        for i in range(0,len(specific_modes)):
            self.assertAlmostEqual(
                specific_amplitudes[specific_modes[i]].item(), 
                all_amplitudes[0,mode_locations[i]],
                places=8
                )