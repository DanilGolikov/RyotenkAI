"""
Shim module: re-exports GPUProviderFactory so that the config validator can import
`src.pipeline.providers.GPUProviderFactory` via importlib.import_module().

The real implementation lives in src.providers.training. Importing this module also
triggers provider auto-registration via src.providers.training.__init__.
"""

from src.providers.training import GPUProviderFactory

__all__ = ["GPUProviderFactory"]
