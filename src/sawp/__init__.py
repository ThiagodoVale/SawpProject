"""SAWP 1.0 scientific analysis package.

The package deliberately exposes a small, stable public surface. Heavy optional
dependencies are imported only by the functions that use them.
"""

from .config import ProjectConfig, load_config
from .domains import Domain

__all__ = ["Domain", "ProjectConfig", "load_config"]
__version__ = "1.0.0"
