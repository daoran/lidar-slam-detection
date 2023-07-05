
try:
    import tensorrt
    from .TensorRT8 import runtime_engine
except Exception as e:
    raise Exception("Runtime backend is unavailable")

RuntimeBackend = runtime_engine.RuntimeBackend