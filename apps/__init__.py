"""Treat apps/ as a package for absolute imports when desired.

Existing scripts often modify sys.path for local runs; this file enables
`import apps.<module>` style imports if repo root is on PYTHONPATH.
"""

