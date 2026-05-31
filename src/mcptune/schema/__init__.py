"""Schema package exports.

Expose the primary schema dataclasses for convenient imports from
``mcptune.schema`` (e.g. ``from mcptune.schema import ToolSpec``).
"""

from .tools import *

__all__ = ["ToolParameter", "ToolSpec"]
