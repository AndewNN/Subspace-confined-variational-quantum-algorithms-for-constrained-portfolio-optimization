"""Make ``Utils`` importable when running scripts from ``experiments/``.

The publication repo is laid out so that ``Utils/`` and ``experiments/`` are
siblings at the project root. When an experiment script is executed with
``cwd=experiments/`` (the convention used throughout the README and the
shell orchestration scripts), ``Utils`` is one directory up and not on the
default ``sys.path``.

Importing this module once at the top of any experiment script puts the
project root on ``sys.path`` so ``from Utils.qaoaCUDAQ import …`` resolves.
The hack is centralised here rather than duplicated in every script.
"""
from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
