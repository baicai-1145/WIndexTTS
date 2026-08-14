"""Deprecated thin wrapper — the webui moved into the package.

Run either:
    python -m windextts.webui        (from source or pip-installed)
    windextts-webui                  (pip-installed console script)
"""
from windextts.webui import main

if __name__ == "__main__":
    main()
