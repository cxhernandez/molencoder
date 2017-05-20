from __future__ import print_function, absolute_import, division

import sys
import argparse

from . import parser_download, parser_train, parser_test

NAME = 'molencoder'
VERSION = '0.1a'


def main():
    help = 'osprey is a tool for machine learning hyperparameter optimization.'
    p = argparse.ArgumentParser(description=help)
    p.add_argument(
        '-V', '--version',
        action='version',
        version='%s %s' % (NAME, VERSION),
    )
    sub_parsers = p.add_subparsers(
        metavar='command',
        dest='cmd',
    )

    parser_download.configure_parser(sub_parsers)
    parser_train.configure_parser(sub_parsers)
    parser_test.configure_parser(sub_parsers)

    if len(sys.argv) == 1:
        sys.argv.append('-h')

    args = p.parse_args()
    args_func(args, p)


def args_func(args, p):
    try:
        args.func(args, p)
    except RuntimeError as e:
        sys.exit("Error: %s" % e)
    except Exception as e:
        if e.__class__.__name__ not in ('ScannerError', 'ParserError'):
            message = """\
An unexpected error has occurred with osprey (version %s), please
consider sending the following traceback to the osprey GitHub issue tracker at:
        https://github.com/cxhernandez/%s/issues
"""
            print(message % (VERSION, NAME), file=sys.stderr)
        raise  # as if we did not catch it
