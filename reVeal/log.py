# -*- coding: utf-8 -*-
"""
log module
"""
import logging
import sys


LOG_FORMAT = logging.Formatter(
    "%(levelname)s - %(asctime)s [%(filename)s:%(lineno)d] : %(message)s"
)


def get_logger(name, log_level=logging.INFO, out_path=None):
    """
    Creates a logger with the specified level, including a stream handler and,
    optionally, a filehandler saved to the specified output path.

    Parameters
    ----------
    name : str
        Name of the logger
    log_level : int, optional
        Log level, by default logging.INFO. Can be specified as an integer (e.g., 20),
        a constant from the logging module (e.g., logging.INFO), or a string
        (e.g., "INFO").
    out_path : pathlib.Path, optional
        If specified, logs will be saved to an output file as well as emitted stdout.
        This can be an file path, in which case outputs will be saved to the specified
        file, or a directory path, in which case the outputs will be saved to a
        log file in the specified directory, with a name of the format
        "<name>.log". Default is None, which does not add a FileHandler.

    Returns
    -------
    logging.Logger
        Logger with stream handler and, optionally, file handler.
    """

    logger = logging.getLogger(name)

    # clear any prior handlers
    logger.handlers.clear()

    if isinstance(log_level, str):
        if log_level not in logging.getLevelNamesMapping():
            raise ValueError(f"Unrecognized value for log_level: {log_level}")
        log_level = logging.getLevelNamesMapping().get(log_level.upper())
    elif isinstance(log_level, int):
        if log_level not in logging.getLevelNamesMapping().values():
            raise ValueError(f"Unrecognized value for log_level: {log_level}")
    else:
        raise TypeError(f"Unrecognized type for log_level: {type(log_level)}")

    # root logger is set to debug
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(LOG_FORMAT)
    stream_handler.setLevel(log_level)
    logger.addHandler(stream_handler)

    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.addHandler(stream_handler)

    if out_path is not None:
        if out_path.is_dir():
            out_log = out_path / f"{name}.log"
        else:
            out_log = out_path

        file_handler = logging.FileHandler(out_log, mode="a")
        file_handler.setFormatter(LOG_FORMAT)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
        warnings_logger.addHandler(file_handler)

    return logger


def remove_streamhandlers(logger):
    """
    Remove StreamHandlers from a logger to stop output to stdout.

    Parameters
    ----------
    logger : logging.Logger
        Logger with StreamHandlers removed
    """

    stream_handlers = [
        h
        for h in logger.handlers
        if isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
    ]
    for stream_handler in stream_handlers:
        logger.removeHandler(stream_handler)


def init_logger(name, log_path=None, verbose=False, node=False):
    """
    Inti

    Parameters
    ----------
    name : str
        Job name; name of log file.
    log_path : str, optional
        If specified, logs will be saved to an output file as well as emitted stdout.
        This can be an file path, in which case outputs will be saved to the specified
        file, or a directory path, in which case the outputs will be saved to a
        log file in the specified directory, with a name of the format
        "<name>.log". Default is None, which does not add a FileHandler.
    verbose : bool, optional
        Option to turn on debug logging.
    node : bool, optional
        Flag for whether this is a node-level logger. If this is a node logger,
        and verbose = False, log_path will be ignored and logs will only be issued
        to stdout.

    Returns
    -------
    logger : logging.Logger
        Logger instance that was initialized
    """

    if verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    if node and not verbose:
        # Node level info loggers only go to STDOUT/STDERR files
        log_path = None

    logger = get_logger(name, log_level=log_level, out_path=log_path)

    return logger
