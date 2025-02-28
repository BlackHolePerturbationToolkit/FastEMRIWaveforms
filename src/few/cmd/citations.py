"""Implementation of the few_citations CLI utility"""

import argparse
import importlib
import sys

from few.utils.citations import Citable


def main():
    parser = argparse.ArgumentParser(
        prog="few_citations",
        description="Export the citations associated to a given module of the FastEMRIWaveforms package",
    )
    parser.add_argument("module")
    args = parser.parse_args(sys.argv[1:])

    few_class: str = args.module

    if not few_class.startswith("few."):
        raise ValueError(
            "The requested class must be part of the 'few' package (e.g. 'few.amplitude.ampinterp2d.AmpInterp2D')."
        )

    module_path, class_name = few_class.rsplit(".", 1)

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise ImportError("Could not import module '{}'.".format(module_path)) from e

    try:
        class_ref = getattr(module, class_name)
    except AttributeError as e:
        raise ImportError(
            "Could not import class '{}' (not found in module '{}')".format(
                class_name, module_path
            )
        ) from e

    if not issubclass(class_ref, Citable):
        print(  # noqa T201
            "Class '{}' ".format(few_class)
            + "does not implement specific references.\n"
            "However, since you are using the FastEMRIWaveform software, "
            "you may cite the following references: \n" + Citable.citation()
        )
        return

    print(class_ref.citation())  # noqa T201


if __name__ == "__main__":
    main()
