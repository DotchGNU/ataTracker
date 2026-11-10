"""DeepLabCut wrappers.

DeepLabCut runs in its own environment (environment-dlc.yml). :mod:`atatracker.dlc.runner` is a
standalone, Python 3.7 compatible script executed there; :mod:`atatracker.dlc.run` builds its
command lines from the configuration. Nothing in this package imports DeepLabCut at import time.
"""
