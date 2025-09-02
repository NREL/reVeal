# # -*- coding: utf-8 -*-
"""reVeal Command Line Interface"""
import logging

from gaps.cli.cli import make_cli

from reVeal import __version__
from reVeal.cli.characterize import characterize_cmd


logger = logging.getLogger(__name__)

commands = [characterize_cmd]
main = make_cli(commands, info={"name": "reVeal", "version": __version__})

if __name__ == "__main__":
    try:
        main(obj={})
    except Exception:
        logger.exception("Error running reVeal CLI")
        raise
