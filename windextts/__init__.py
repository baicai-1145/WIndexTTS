
from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("windextts")
except PackageNotFoundError:  # source checkout without install
    __version__ = "0.2.0"
