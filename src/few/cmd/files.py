"""Implementation of the few_files CLI utility"""

import argparse
import logging
import sys
import typing

from few.utils.globals import Globals, get_file_manager, get_logger


def few_files_fetch(args: argparse.Namespace):
    """Run the 'few_files fetch' subcommand."""
    file_manager = get_file_manager()

    args = vars(args)

    logger = get_logger()

    if logger.level > logging.INFO:
        logger.setLevel(logging.INFO)

    if "list_tags" in args and args["list_tags"]:
        logger.info("Available tags: {}".format(file_manager.get_tags()))
        return

    tag = args["tag"] if "tag" in args else None

    if tag is None:
        logger.info(
            "Downloading all missing files into '{}'\n".format(
                file_manager.download_dir
            )
        )
        file_manager.prefetch_all_files(
            discarded_tags=["deprecated"], skip_disabled=args["skip_disabled"]
        )
    else:
        logger.info(
            "Downloading all missing files tagged '{}' into '{}'\n".format(
                tag, file_manager.download_dir
            )
        )
        file_manager.prefetch_files_by_tag(tag=tag, skip_disabled=args["skip_disabled"])
    logger.info("\nDone.")


def _few_files_fetch(subparsers):
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "fetch",
        help="Pre-fetch files of given tag (or all known files if no tag is provided)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--tag", dest="tag", help="Tag of files to fetch", default=None)
    group.add_argument(
        "--list-tags",
        dest="list_tags",
        action="store_true",
        help="Print list of available tags",
    )
    parser.add_argument("--skip-disabled", dest="skip_disabled", action="store_true")

    parser.set_defaults(callback=few_files_fetch)


def few_files_list(args: argparse.Namespace):
    """Run the 'few_files list' subcommand."""
    file_manager = get_file_manager()
    logger = get_logger()
    if logger.level > logging.INFO:
        logger.setLevel(logging.INFO)

    file_manager.build_local_cache()

    logger.info("Listing files from the File Registry:")
    for file in file_manager.registry.files:
        path = file_manager.try_get_local_file(file.name, use_cache=True)
        logger.info(
            "  - {}: {}".format(
                file.name,
                f"found in '{path.parent}'" if path is not None else "not found",
            )
        )


def _few_files_list(subparsers):
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "list", help="Locate and list files"
    )
    parser.set_defaults(callback=few_files_list)


def _few_files_parser(
    parent_parsers: typing.Sequence[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Build the CLI argument parser"""
    parser = argparse.ArgumentParser(prog="few_files", parents=parent_parsers)
    subparsers = parser.add_subparsers()
    _few_files_fetch(subparsers)
    _few_files_list(subparsers)
    return parser


def main():
    globals = Globals()
    globals.init(cli_args=sys.argv[1:])
    _, _, extra_args = globals.config.get_extras()
    parent_parsers = globals.config.build_cli_parent_parsers()
    args = _few_files_parser(parent_parsers).parse_args(extra_args)
    args.callback(args)


if __name__ == "__main__":
    main()
