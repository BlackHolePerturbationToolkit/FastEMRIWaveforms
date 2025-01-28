"""Implementation of few_files CLI utility"""

import argparse
import sys

from ..utils.globals import Globals, get_logger


def few_files_fetch(args):
    """Run the 'few_files fetch' subcommand."""
    get_logger().warning("running few_files fetch subcommand.")


def _few_files_fetch(subparsers):
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "fetch", help="Pre-fetch files"
    )
    parser.add_argument("--tag", help="Tag of files to fetch")
    parser.set_defaults(callback=few_files_fetch)


def few_files_list(args):
    """Run the 'few_files list' subcommand."""
    get_logger().warning("running few_files list subcommand.")


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
    Globals().init(cli_args=sys.argv[1:])
    parser = _few_files_parser()
    extra_args = Globals().config._extra_cli
    args, _ = parser.parse_known_args(extra_args)
    args.callback(args)
