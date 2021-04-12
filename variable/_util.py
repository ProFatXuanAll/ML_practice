from typing import Any
from variable._var import Variable


def coerce(value: Any) -> Variable:
    r"""Coerce value into ``Variable``."""
    if isinstance(value, Variable):
        return value
    elif isinstance(value, (float, int)):
        return Variable(value)
    else:
        raise TypeError(f'{type(value)} is not compatible with ``Variable``.')
