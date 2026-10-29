#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Output-variant switches shared by the plotting scripts.

The appendix carries a second copy of every algorithm comparison with the
differential-privacy variant PDFFed-DP removed, so that the reader can see what
the non-private method does on its own.  Rather than repeat the same flag in
seven scripts, each one asks this module which algorithms to draw and whether
to tag its output file name.

Two ways to switch it on::

    python analysis/plot_communication.py --no-dp
    set FL_HIDE_ALGOS=PDFFed_DP && python analysis/plot_communication.py

The environment variable takes a comma-separated list, so other algorithms can
be hidden the same way.  With PDFFed-DP hidden the figures are written with a
``_noDP`` suffix, which leaves the full versions untouched.
"""

import os
import sys

ENV = "FL_HIDE_ALGOS"
OWN_DP = "PDFFed_DP"


def hidden():
    """Names to leave out of the figures, from the env var and ``--no-dp``."""
    names = {x.strip() for x in os.environ.get(ENV, "").split(",") if x.strip()}
    if "--no-dp" in sys.argv:
        names.add(OWN_DP)
    return names


def keep(name):
    """True when ``name`` should still be drawn."""
    return name not in hidden()


def active(names):
    """The subset of ``names`` that survives the current switches."""
    return tuple(n for n in names if keep(n))


def suffix():
    """File-name tag for the current switches (empty in the default variant)."""
    return "_noDP" if OWN_DP in hidden() else ""
