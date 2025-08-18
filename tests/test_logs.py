# -*- coding: utf-8 -*-
"""
logs module tests
"""
import logging
from pathlib import Path

import pytest

from loci.logs import get_logger


def test_get_logger_no_filehandler(caplog):
    """ "
    Unit test for get_logger() without a FileHandler
    """
    logger_name = "test-log-no-file"
    logger = get_logger(logger_name, log_level=logging.INFO)

    msgs = [
        ("Hello!", logging.INFO),
        ("Beware!", logging.WARNING),
        ("Trouble!", logging.ERROR),
    ]
    for msg, level in msgs:
        logger.log(level, msg)

    for i, record_tuple in enumerate(caplog.record_tuples):
        lname, level, msg = record_tuple
        assert lname == logger_name
        assert msg == msgs[i][0]
        assert level == msgs[i][1]

    del logger


def test_get_logger_filehandler_dirpath(caplog, tmp_path):
    """ "
    Unit test for get_logger() with a FileHandler specified as a directory path
    """
    logger_name = "test-log-dirpath"
    logger = get_logger(logger_name, log_level=logging.INFO, out_path=tmp_path)

    msgs = [
        ("Hello!", logging.INFO),
        ("Beware!", logging.WARNING),
        ("Trouble!", logging.ERROR),
    ]
    for msg, level in msgs:
        logger.log(level, msg)

    for i, record_tuple in enumerate(caplog.record_tuples):
        lname, level, msg = record_tuple
        assert lname == logger_name
        assert msg == msgs[i][0]
        assert level == msgs[i][1]

    log_file = logger.handlers[1].baseFilename
    assert Path(log_file).exists()
    with open(log_file, "r") as f:
        log_lines = f.readlines()

    for i, log_line in enumerate(log_lines):
        assert log_line.startswith(logging.getLevelName(msgs[i][1]))
        assert log_line.endswith(f"{msgs[i][0]}\n")


def test_get_logger_filehandler_filepath(caplog, tmp_path):
    """ "
    Unit test for get_logger() with a FileHandler specified as a directory path
    """
    logger_name = "test-log-filepath"
    log_file = tmp_path.joinpath("file.log")
    logger = get_logger(logger_name, log_level=logging.INFO, out_path=log_file)

    msgs = [
        ("Hello!", logging.INFO),
        ("Beware!", logging.WARNING),
        ("Trouble!", logging.ERROR),
    ]
    for msg, level in msgs:
        logger.log(level, msg)

    for i, record_tuple in enumerate(caplog.record_tuples):
        lname, level, msg = record_tuple
        assert lname == logger_name
        assert msg == msgs[i][0]
        assert level == msgs[i][1]

    assert Path(log_file).exists()
    with open(log_file, "r") as f:
        log_lines = f.readlines()

    for i, log_line in enumerate(log_lines):
        assert log_line.startswith(logging.getLevelName(msgs[i][1]))
        assert log_line.endswith(f"{msgs[i][0]}\n")


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
