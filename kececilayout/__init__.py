# Fixes F401: '.kececilayout' imported but unused by making the import conditional
# Add necessary blank lines to satisfy E302 and E305

from __future__ import annotations
import importlib
import warnings
from .kececi_layout import *


def eski_fonksiyon():
    warnings.warn("Keçeci Layout; Python 3.7-3.14 sürümlerinde sorunsuz çalışmalıdır.", DeprecationWarning)
eski_fonksiyon()

__version__ = "0.2.0"
