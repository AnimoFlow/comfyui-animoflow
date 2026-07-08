"""Repo-root pytest config.

The repo root is a ComfyUI custom-node package (``__init__.py`` at the
top level, imported by ComfyUI with package context). pytest must not
try to import that ``__init__.py`` outside a package — keep it out of
collection and let the tests do their own flat-path imports.
"""
collect_ignore = ["__init__.py", "nodes", "containers", "comfyui"]
