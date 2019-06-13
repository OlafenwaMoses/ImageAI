import os

_BACKEND = "tensorflow"

if "KERAS_BACKEND" in os.environ:
    _backend = os.environ["KERAS_BACKEND"]

    backends = {
        "tensorflow"
    }

    assert _backend in backends

    _BACKEND = _backend

if _BACKEND == "tensorflow":
    from .tensorflow_backend import *
else:
    raise ValueError("Unknown backend: " + str(_BACKEND))
