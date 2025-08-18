# -*- coding: utf-8 -*-
"""
logs module
"""
import logging
import datetime

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
        Log level, by default logging.INFO.
    out_path : pathlib.Path, optional
        If specified, logs will be saved to an output file as well as emitted stdout.
        This can be an file path, in which case outputs will be saved to the specified
        file, or a directory path, in which case the outputs will be saved to a
        log file in the specified directory, with a name of the format
        "<name>_<year>-<month>-<day>_<hour><minute><second>.log". Default is None,
        which does not add a FileHandler.

    Returns
    -------
    logging.Logger
        Logger with stream handler and, optionally, file handler.
    """

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(LOG_FORMAT)
    stream_handler.setLevel(log_level)
    logger.addHandler(stream_handler)

    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.addHandler(stream_handler)

    if out_path is not None:
        if out_path.is_dir():
            now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
            out_log = out_path.joinpath(f"{name}-{now}.log")
        else:
            out_log = out_path

        file_handler = logging.FileHandler(out_log, mode="w")
        file_handler.setFormatter(LOG_FORMAT)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
        warnings_logger.addHandler(file_handler)

    return logger
