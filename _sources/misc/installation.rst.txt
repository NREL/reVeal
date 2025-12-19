.. _installation:

Installation
============

1. Clone the repository

    .. code-block:: shell

        git clone git@github.com:NREL/reVeal.git

2. Move into the local repository

    .. code-block:: shell

        cd reVeal


3. Recommended: Setup virtual environment with `conda`/`mamba`:

    .. code-block:: shell

        mamba env create -f environment.yml
        mamba activate reVeal

    Note: You may choose an alternative virtual environment solution; however, installation of dependencies is not guaranteed to work.

4. Install ``reVeal``:
    .. code-block:: shell

        pip install .


Usage
==================
For details on usage of command line tools, see: :doc:`../_cli/cli`.
