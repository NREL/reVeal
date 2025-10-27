# # -*- coding: utf-8 -*-
"""reVeal Command Line Interface"""
import logging

from gaps.cli.cli import make_cli

from reVeal import __version__
from reVeal.cli.characterize import characterize_cmd
from reVeal.cli.normalize import normalize_cmd
from reVeal.cli.score_weighted import score_weighted_cmd
from reVeal.cli.downscale import downscale_cmd

logger = logging.getLogger(__name__)

commands = [characterize_cmd, normalize_cmd, score_weighted_cmd, downscale_cmd]
main = make_cli(commands, info={"name": "reVeal", "version": __version__})

# export GAPs commands to namespace for documentation
batch = main.commands["batch"]
pipeline = main.commands["pipeline"]
script = main.commands["script"]
status = main.commands["status"]
reset_status = main.commands["reset-status"]
template_configs = main.commands["template-configs"]

if __name__ == "__main__":
    try:
        main(obj={})
    except Exception:
        logger.exception("Error running reVeal CLI")
        raise
