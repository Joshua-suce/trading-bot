import inspect
from typing import Any


def generate_with_context(generator: Any, frame: Any, higher_trend_bias: int):
    method = generator.generate
    parameters = inspect.signature(method).parameters.values()
    accepts_context = any(
        parameter.name == "higher_trend_bias"
        or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    if accepts_context:
        return method(frame, higher_trend_bias=higher_trend_bias)
    return method(frame)
