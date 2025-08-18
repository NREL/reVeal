# # -*- coding: utf-8 -*-
"""loci Command Line Interface"""
import logging

from gaps.cli.cli import make_cli

from loci import __version__
from loci.cli.characterize import characterize_cmd


logger = logging.getLogger(__name__)

commands = [characterize_cmd]
main = make_cli(commands, info={"name": "loci", "version": __version__})

if __name__ == "__main__":
    try:
        main(obj={})
    except Exception:
        logger.exception("Error running loci CLI")
        raise
