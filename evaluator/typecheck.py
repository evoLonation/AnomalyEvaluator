from typing import overload
from jaxtyping import Float, Bool, Int, jaxtyped
from beartype import beartype
from beartype.door import die_if_unbearable

typechecker = jaxtyped(typechecker=beartype)

@overload
def check_type[T](value, type: type[T]) -> T: ...

@overload
def check_type[T](value, type, array: type[T], shape: str) -> T: ...

def check_type(value, type, array = None, shape = None):
    if array is not None:
        die_if_unbearable(value, type[array, shape])
        return value
    else:
        die_if_unbearable(value, type)
        return value