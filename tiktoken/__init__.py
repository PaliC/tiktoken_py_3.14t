# This is the public API of tiktoken
from .core import Encoding as Encoding
from .model import encoding_for_model as encoding_for_model
from .model import encoding_name_for_model as encoding_name_for_model
from .registry import get_encoding as get_encoding
from .registry import list_encoding_names as list_encoding_names
import sys
import warnings

__version__ = "0.12.0"

# Check for free-threaded mode
try:
    GIL_ENABLED = sys._is_gil_enabled()
except AttributeError:
    GIL_ENABLED = True

if GIL_ENABLED:
    warnings.warn(
        "Running with GIL enabled. For best performance, use Python 3.14t:\n"
        "  Build Python with: ./configure --disable-gil\n"
        "  Or set: PYTHON_GIL=0",
        RuntimeWarning,
        stacklevel=2
    )


__all__ = ['GIL_ENABLED']