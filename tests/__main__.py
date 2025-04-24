import argparse
import dataclasses
import sys
import typing as t
from unittest import TestLoader, TestProgram


class TestFew(TestProgram):
    def createTests(self, from_discovery=False, Loader=None):
        import pathlib

        current_directory = pathlib.Path(__file__).parent
        self.test = TestLoader().discover(start_dir=current_directory)


@dataclasses.dataclass
class TestFewOpts:
    disabled_tags: t.Optional[list[str]] = None


def process_argv(argv: t.Optional[list[str]] = None) -> tuple[TestFewOpts, list[str]]:
    argv = sys.argv if argv is None else argv

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--disable", help="Tags to disable", dest="disabled_tags", action="append"
    )
    parser.add_argument("-H", action="help")
    parsed = parser.parse_known_args(argv)

    opts = TestFewOpts(**vars(parsed[0]))
    return opts, parsed[1]


if __name__ == "__main__":
    import logging

    from few import get_config, get_config_setter, get_file_manager, get_logger

    config_setter = get_config_setter(reset=True)

    options, argv = process_argv(sys.argv)

    if options.disabled_tags is not None:
        config_setter.disable_file_tags(*options.disabled_tags)

    if get_logger().getEffectiveLevel() > logging.INFO:
        get_config_setter(reset=True).set_log_level("INFO")

    if get_config().file_allow_download:
        get_logger().info("Ensuring that all files required by tests are present.")
        get_file_manager().prefetch_files_by_tag("testfile", skip_disabled=True)
        get_logger().info("Done... Now run the tests!")

    config_setter = get_config_setter(reset=True)

    if options.disabled_tags is not None:
        config_setter.disable_file_tags(*options.disabled_tags)

    TestFew(argv=argv)
