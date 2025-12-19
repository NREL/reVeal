.. toctree::
   :hidden:

   Home page <self>
   Installation and Usage <misc/installation_usage>
   Examples <misc/examples>
   CLI reference <_cli/cli>
   API reference <_autosummary/reVeal>

reVeal documentation
********************

What is reVeal?
===============

``reVeal`` (the reV Extension for Analyzing Large Loads) is an open-source geospatial software package for modeling the site-suitability and spatial patterns of deployment of large sources of electricity demand under future scenarios. ``reVeal`` is part of the `reV ecosystem of tools <https://nrel.github.io/reV/#rev-ecosystem>`_.

How does reVeal work?
=====================
``reVeal`` consists of a library of Python classes and functions which are wrapped in a series of CLI commands. These commands can be run individually or as composed pipelines to enable model execution.

``reVeal`` is typically run on a user-input geospatial (GIS) grid covering the the user's area of interest. The modeling process typically follows the following workflow:

#. Characterize each grid cell, according to a number of other GIS datasets. This step overlays the grid with other spatial datasets and attributes each grid cell with values derived from those datasets.
#. Normalize the vales from the prior step, such that all values are on the same scale from 0 to 1.
#. Calculate a composite site-suitability score for each grid cell, based on a user-specified weighted-criteria model of normalized characterization values.
#. Downscale user-input projections of large load growth to grid cells, based on site-suitability and other user-specified settings and constraints.

`reVeal` is designed to be compatible for use in both a high-performance computing (HPC) environment, or on a local machine. This enables flexibility allows easy set up for users with standard hardware, as well as execution at high-resolution and large spatial scales for users with access to HPC.
