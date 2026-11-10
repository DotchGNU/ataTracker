"""DeepLabCut wrappers.

DeepLabCut runs in its own environment (environment-dlc.yml). :mod:`ataagent.dlc.runner` is a
standalone, Python 3.7 compatible script executed there; :mod:`ataagent.dlc.run` builds its
command lines from the configuration. Nothing in this package imports DeepLabCut at import time.
"""
