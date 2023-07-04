from hardware.platform_common import JETPACK

if JETPACK == "5.0.2":
    from .TensorRT8_4 import runtime_engine

RuntimeBackend = runtime_engine.RuntimeBackend