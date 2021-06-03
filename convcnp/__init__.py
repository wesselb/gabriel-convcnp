# noinspection PyUnresolvedReferences
import lab.torch  # Load PyTorch extension.
import plum

_dispatch = plum.Dispatcher()

from .data import *
from .encoder import *
from .decoder import *
from .discretisation import *
from .unet import *
from .unet import *
from .convcnp import *
