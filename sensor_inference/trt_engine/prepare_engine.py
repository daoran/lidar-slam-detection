from hardware.platform_common import JETPACK

if JETPACK == "5.0.2":
    from .TensorRT8_4 import prepare_engine

PrepareBackend = prepare_engine.PrepareBackend