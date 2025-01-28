"""Implementation of few_files CLI utility"""

import argparse
import logging
import sys

from ..utils.globals import Globals, get_logger, get_file_manager


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
        file_manager.prefetch_all_files()
    else:
        logger.info(
            "Downloading all missing files tagged '{}' into '{}'\n".format(
                tag, file_manager.download_dir
            )
        )
        file_manager.prefetch_files_by_tag(tag=tag)
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

    parser.set_defaults(callback=few_files_fetch)


def few_files_list(args: argparse.Namespace):
    """Run the 'few_files list' subcommand."""


def _few_files_list(subparsers):
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "list", help="Locate and list files"
    )
    parser.set_defaults(callback=few_files_list)


def _few_files_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser"""
    parser = argparse.ArgumentParser(prog="few_files")
    subparsers = parser.add_subparsers()
    _few_files_fetch(subparsers)
    _few_files_list(subparsers)
    return parser


def few_files():
    globals = Globals()
    globals.init(cli_args=sys.argv[1:])
    _, _, extra_args = globals.config.get_extras()
    args, _ = _few_files_parser().parse_known_args(extra_args)
    args.callback(args)
