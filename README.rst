******
reVeal
******

|License| |Zenodo| |Ruff| |Pixi| |SWR| |Codecov|

.. |PythonV| image:: https://badge.fury.io/py/reVeal.svg
    :target: https://pypi.org/project/reVeal/

.. |PyPi| image:: https://img.shields.io/pypi/pyversions/reVeal.svg
    :target: https://pypi.org/project/reVeal/

.. |Ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
    :target: https://github.com/astral-sh/ruff

.. |License| image:: https://img.shields.io/badge/License-BSD_3--Clause-orange.svg
    :target: https://opensource.org/licenses/BSD-3-Clause

.. |Pixi| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json
    :target: https://pixi.sh

.. |SWR| image:: https://img.shields.io/badge/SWR--25--147_-blue?label=NLR
    :alt: Static Badge

.. |Zenodo| image:: https://zenodo.org/badge/1038758980.svg
    :target: https://doi.org/10.5281/zenodo.17984036

.. |Codecov| image:: https://codecov.io/github/NREL/reVeal/graph/badge.svg?token=NXBPNUPT3Y
    :target: https://codecov.io/github/NREL/reVeal

.. inclusion-intro


``reVeal`` (the reV Extension for Analyzing Large Loads) is an open-source geospatial
software package for modeling the site-suitability and spatial patterns of deployment
of large sources of electricity demand under future scenarios. ``reVeal`` is part of
the `reV ecosystem of tools <https://nrel.github.io/reV/#rev-ecosystem>`_.


Installing reVeal
=================
The quickest way to install reVeal for users is from PyPI:

.. code-block:: bash

    pip install NLR-reVeal

If you would like to install and run reVeal from source, we recommend using `pixi <https://pixi.sh/latest/>`_:

.. code-block:: bash

    git clone git@github.com:NREL/reVeal.git; cd reVeal
    pixi run reVeal

For detailed instructions, see the `installation documentation <https://nrel.github.io/reVeal/misc/installation.html>`_.


Quickstart
==========
.. To run a quick reVeal demo, use:

.. .. code-block:: shell

..     pixi run demo

.. This will generate sample map outputs using example reV geothermal supply curve outputs.

For more information on running ``reVeal``, see
`Usage <https://github.com/NREL/reVeal/blob/main/USAGE.md>`_.


Development
===========
Please see the `Development Guidelines <https://nrel.github.io/reVeal/dev/index.html>`_
if you wish to contribute code to this repository.
