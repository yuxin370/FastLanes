# ────────────────────────────────────────────────────────
# |                      FastLanes                       |
# ────────────────────────────────────────────────────────
# python/pyfastlanes/__init__.py
# ────────────────────────────────────────────────────────
"""
PyFastLanes Python API
"""

try:
    from ._fastlanes import (
        get_version,
        Connection,
        connect,
        JpegLoader,
        ImageHeader,
        ColorSpace,
        QuantTable,
        ChannelDCT,
        DCTBlockRow,
        # Add any other bindings you expose here
    )
except ImportError:  # pragma: no cover - fallback for missing extension
    def get_version() -> str:
        """Return a placeholder version when the C++ extension is unavailable."""

        return "0.0.0"

    class Connection:  # type: ignore
        """Dummy placeholder for the native :class:`Connection` class."""

        def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - runtime
            raise ImportError(
                "pyfastlanes C++ extension is not built; install from source "
                "or build the extension to use this functionality"
            )

    # --- JPEG Loader fallbacks ---
    class JpegLoader:
        """Dummy placeholder for JpegLoader."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "JpegLoader requires the C++ extension to be built."
            )

        @staticmethod
        def load_header(path: str):
            raise ImportError("JpegLoader.load_header requires the C++ extension.")

    # Dummy classes (minimal placeholders)
    class ImageHeader: pass
    class ColorSpace: pass
    class QuantTable: pass
    class ChannelDCT: pass
    class DCTBlockRow: pass

    def connect() -> Connection:
        """Return a :class:`Connection` instance using the fallback."""

        return Connection()

__all__ = [
    "get_version",
    "Connection",
    "connect",
    "JpegLoader",
    "ImageHeader",
    "ColorSpace",
    "QuantTable",
    "ChannelDCT",
    "DCTBlockRow",
    # Add others as needed
]
