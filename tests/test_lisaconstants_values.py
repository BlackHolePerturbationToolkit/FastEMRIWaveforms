"""FEW's native constants come from lisaconstants (2026-10-08).

``src/few/cutils/global.h`` reads the generated ``lisaconstants_values.h``
(written by LAT's ``lisatools.utils.lisaconstants_header``; FEW does not
depend on LAT, so this test re-derives every value from lisaconstants itself).
"""
import os
import re
import unittest

import lisaconstants as lc

HEADER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "src", "few", "cutils", "lisaconstants_values.h")


def _defines(text):
    pat = re.compile(r"^#define\s+LISACONSTANTS_(\w+)\s+([-+0-9.eE]+)")
    return {m.group(1): float(m.group(2))
            for m in (pat.match(line.strip()) for line in text.splitlines()) if m}


class LisaconstantsHeaderTest(unittest.TestCase):
    def test_every_value_is_lisaconstants(self):
        with open(HEADER) as fh:
            got = _defines(fh.read())
        derived = {
            "MTSUN": lc.SOLAR_MASS_PARAMETER / lc.SPEED_OF_LIGHT**3,
            "MRSUN": lc.SOLAR_MASS_PARAMETER / lc.SPEED_OF_LIGHT**2,
            "AU_LIGHT_TIME": lc.ASTRONOMICAL_UNIT / lc.SPEED_OF_LIGHT,
            "GPC_LIGHT_TIME": 1e9 * lc.PARSEC / lc.SPEED_OF_LIGHT,
        }
        for name, val in got.items():
            want = derived[name] if name in derived else float(getattr(lc, name))
            self.assertEqual(val, want, name)
        for name in ("ASTRONOMICAL_YEAR", "MTSUN", "AU_LIGHT_TIME", "GPC_LIGHT_TIME"):
            self.assertIn(name, got)        # the four global.h uses


if __name__ == "__main__":
    unittest.main()
