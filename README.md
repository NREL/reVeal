# reVeal
The reV Extension for Analyzing Large Loads.

## Installation
1. Clone the repo
    ```commandline
    git clone git@github.com:NREL/reVeal.git
    ```

2. Move into the local repo
    ```command line
    cd reVeal
    ```

3. Recommended: Setup virtual environment with `conda`/`mamba`:
    ```commandline
    mamba env create -f environment.yml
    mamba activate reVeal
    ```
    Note: You may choose an alternative virtual environment solution; however, installation of dependencies is not guaranteed to work.

4. Install `reVeal`:
    - For users: `pip install .`
    - For developers: `pip install -e '.[dev]'`

5. **Developers Only** Install pre-commit
```commandline
pre-commit install
```

## Usage
Refer to the [Usage](USAGE.md) documentation.
