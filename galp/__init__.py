"""GALP Python APIs.

The native extension remains an implementation detail.  Applications should
import stable readers from :mod:`galp.torch` and semantic profiles from
:mod:`galp.profiles`.
"""

from .profiles import DirectDctProfile

__all__ = ["DirectDctProfile"]
