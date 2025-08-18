# -*- coding: utf-8 -*-
"""
log module tests
"""
import logging

import pytest

from loci.log import get_logger, remove_streamhandlers, init_logger


@pytest.mark.parametrize(
    "log_level",
    [
        "INFO",
        logging.INFO,
        20,
        "DEBUG",
        logging.DEBUG,
        10,
    ],
)
@pytest.mark.parametrize(
    "to_file",
    [
        True,
        False,
    ],
)
def test_get_logger(capsys, caplog, tmp_path, log_level, to_file):
    """
    Unit tests for get_logger() function. Check that it handles setting of log level
    and option to write to file.
    """

    if to_file:
        out_path = tmp_path
    else:
        out_path = None

    logger_name = "test-log"
    logger = get_logger(logger_name, log_level=log_level, out_path=out_path)

    msgs = [
        ("Hello!", logging.INFO),
        ("Beware!", logging.WARNING),
        ("Trouble!", logging.ERROR),
        ("Debug Me!", logging.DEBUG),
    ]
    for msg, level in msgs:
        logger.log(level, msg)

    # check root logger caught all messages (root logger is always at debug level)
    for i, record_tuple in enumerate(caplog.record_tuples):
        lname, level, msg = record_tuple
        assert lname == logger_name
        assert msg == msgs[i][0]
        assert level == msgs[i][1]

    captured_stdout = capsys.readouterr().out
    if len(captured_stdout) > 0:
        captured_messages = captured_stdout.split("\n")[:-1]
    else:
        captured_messages = []

    # check that correct messages were logged to stdout based on the log_level
    if log_level in ("INFO", logging.INFO):
        expected_messages = list(zip(*msgs))[0][:-1]
    elif log_level in ("DEBUG", logging.DEBUG):
        expected_messages = list(zip(*msgs))[0]
    else:
        raise NotImplementedError(f"No check implemented for log_level {log_level}")

    assert len(captured_messages) == len(
        expected_messages
    ), "Unexpected number of messages logged to stdout"
    for i, message in enumerate(expected_messages):
        assert (
            message in captured_messages[i]
        ), f"Captured message {i+1} does not contain expected text {message}"

    if out_path:
        expected_logfile = tmp_path / f"{logger_name}.log"
        assert expected_logfile.exists(), "Output file not created"
        with open(expected_logfile, "r") as f:
            log_lines = f.readlines()
            for i, message in enumerate(expected_messages):
                assert (
                    message in log_lines[i]
                ), f"Log file line {i+1} does not contain expected text {message}"


@pytest.mark.parametrize(
    "log_level,exception", [("ALL", ValueError), (99, ValueError), ([10], TypeError)]
)
def test_get_logger_level_error(log_level, exception):
    """
    Unit test for get_logger() - tests that errors are raised or bad input values for
    log_level.
    """

    logger_name = "test-log"
    if exception == ValueError:
        err = "Unrecognized value for log_level: *."
    elif exception == TypeError:
        err = "Unrecognized type for log_level: *."
    else:
        raise NotImplementedError(f"No check implemented for exception {exception}")

    with pytest.raises(exception, match=err):
        get_logger(logger_name, log_level=log_level, out_path=None)


def test_remove_streamhandlers(capsys):
    """
    Unit test for remove_streamhandlers().
    """

    logger_name = "test-log"
    logger = get_logger(logger_name, log_level="INFO", out_path=None)

    assert len(logger.handlers) == 1, "Logger was not initialized with handlers"

    logger.info("Message 1")
    remove_streamhandlers(logger)

    assert len(logger.handlers) == 0, "Logger still has handlers"

    logger.info("Message 2")

    captured_stdout = capsys.readouterr().out

    assert (
        "Message 2" not in captured_stdout
    ), "Message was issued to stdout after removing streamhandler"


@pytest.mark.parametrize("verbose", [True, False])
@pytest.mark.parametrize("node", [True, False])
@pytest.mark.parametrize("to_file", [True, False])
def test_init_logger(caplog, capsys, tmp_path, verbose, node, to_file):
    """ "
    Unit test for init_logger() under various parameters
    """
    if to_file:
        out_path = tmp_path
    else:
        out_path = None

    logger_name = "test-log"
    logger = init_logger(logger_name, out_path, verbose, node)
    msgs = [
        ("Hello!", logging.INFO),
        ("Beware!", logging.WARNING),
        ("Trouble!", logging.ERROR),
        ("Debug Me!", logging.DEBUG),
    ]
    for msg, level in msgs:
        logger.log(level, msg)

    # check root logger caught all messages (root logger is always at debug level)
    for i, record_tuple in enumerate(caplog.record_tuples):
        lname, level, msg = record_tuple
        assert lname == logger_name
        assert msg == msgs[i][0]
        assert level == msgs[i][1]

    captured_stdout = capsys.readouterr()[0]
    if len(captured_stdout) > 0:
        captured_messages = captured_stdout.split("\n")[:-1]
    else:
        captured_messages = []

    # check verbose is handled correctly
    if verbose:
        expected_messages = list(zip(*msgs))[0]
    else:
        expected_messages = list(zip(*msgs))[0][:-1]

    assert len(captured_messages) == len(
        expected_messages
    ), "Unexpected number of messages logged to stdout"
    for i, message in enumerate(expected_messages):
        assert (
            message in captured_messages[i]
        ), f"Captured message {i+1} does not contain expected text {message}"

    # check to_file is handled correctly
    expected_logfile = tmp_path / f"{logger_name}.log"
    if (node and not verbose) or not to_file:
        assert not expected_logfile.exists(), "Output file created"
    else:
        assert expected_logfile.exists(), "Output file not created"


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
