try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version  # For Python <3.8

__version__ = "0.0.1"


from .tts import SwarlekhaTTS
from .vc import SwarlekhaVC
