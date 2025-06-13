"""Tomogram reconstruction, subtomogram reconstruction, and subtilt extraction for cryo-ET."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-tomogram")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Talley Lambert"
__email__ = "talley.lambert@gmail.com"
