# -*- coding: utf-8 -*-
"""
dataframe module
"""
import numpy as np


def dataframe_split(df, n_parts):
    """
    Splits dataframe into the specified number of roughly equal parts. This function
    is similar to numpy.array_split, with a few major differences:
    1. It expects an input dataframe, not an array.
    2. It only accepts as input the number of input parts (am array of indices cannot
       be passed)
    3. It does not use .swapaxes, which is deprecated and raises a FutureWarning.
    4. The number of parts returned will be capped at the size of the input dataframe,
       whereas numpy.array_split always returns exactly n_parts, even if some parts are
       empty.
    5. It returns a generator, not a list.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe to split
    n_parts : int
        Number of desired parts

    Yields
    ------
    pandas.DataFrame
        Yields parts of split dataframe.

    Raises
    ------
    ValueError
        A ValueError will be raised if the numer of input sections is < 1.
    """

    n_rows = len(df)
    # cap the number of sections at the total size of the dataframe
    n_sections = np.minimum(n_rows, int(n_parts))
    if n_sections <= 0:
        raise ValueError("number sections must be larger than 0.") from None
    nrows_per_section, extras = divmod(n_rows, n_sections)
    section_sizes = (
        [0]
        + extras * [nrows_per_section + 1]
        + (n_sections - extras) * [nrows_per_section]
    )
    div_points = np.array(section_sizes, dtype=np.int32).cumsum()

    for i in range(n_sections):
        start_row = div_points[i]
        end_row = div_points[i + 1]
        yield df.iloc[start_row:end_row]
