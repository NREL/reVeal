# -*- coding: utf-8 -*-
"""
errors module
"""


class CSVReadError(Exception):
    """
    Exception for errors generally related to failed CSV reads.
    """


class FileFormatError(Exception):
    """
    Exception for unexpected file format when reading an input file.
    """
