import inspect
import re
from typing import Any, List, Optional, Tuple, Union

DottyPath = Union[str, List, Tuple]

_MISSING = object()


def is_int(value: Any) -> bool:
    """Check if the value is an integer or can be converted to an integer."""
    if isinstance(value, int):
        return True
    if isinstance(value, str):
        return value.isdigit() or (value.startswith("-") and value[1:].isdigit())
    return False


def path_to_string(path: DottyPath) -> str:
    """Serialise a path string or list/tuple into a dot notation string."""
    if isinstance(path, (list, tuple)):
        return ".".join(str(part) for part in path)

    if not path:
        return ""

    parts = parse_path(path)
    return ".".join(str(part) for part in parts)


def parse_path(path: DottyPath) -> List[Union[str, int]]:
    """Parse a path string or list/tuple into a list of path segments."""
    if isinstance(path, (list, tuple)):
        return list(path)

    if not path:
        return []

    result: List[Union[str, int]]

    # Handle bracket notation like 'foo[0][bar]'
    if "[" in path:
        parts = re.findall(r"([^\.\[\]]+|\[\d+\]|\[[^\[\]]+\])", path)
        result = []
        for part in parts:
            if part.startswith("[") and part.endswith("]"):
                content = part[1:-1]
                if is_int(content):
                    result.append(int(content))
                else:
                    result.append(content)
            else:
                result.append(part)
        return result

    # Handle dot notation like 'foo.bar' or 'foo.0.bar'
    result = []
    for part in path.split("."):
        if is_int(part):
            result.append(int(part))
        else:
            result.append(part)
    return result


def _detect_callable_type(fn: Any) -> str:
    """
    Detect the type of callable to determine how to invoke it properly.

    Returns:
      String describing the callable type: "bound_method", "unbound_method",
      "class_constructor", "function", or "callable"
    """
    # Check if it's a bound method (has __self__ that is not None)
    if hasattr(fn, "__self__") and fn.__self__ is not None:
        return "bound_method"

    # Check for method objects (bound methods)
    if inspect.ismethod(fn):
        return "bound_method"

    # Check if it's a class constructor
    if inspect.isclass(fn):
        return "class_constructor"

    # Check if it's a regular function or builtin
    if inspect.isfunction(fn) or inspect.isbuiltin(fn):
        # Only consider it an unbound method if:
        # 1. It's a regular function (not builtin)
        # 2. It has a __qualname__ with a dot (indicating it's defined in a class)
        # 3. It doesn't have __self__ (which would make it bound)
        # 4. The first parameter is named 'self' or 'cls' (indicating it's an instance/class method)
        if (
            inspect.isfunction(fn)
            and hasattr(fn, "__qualname__")
            and "." in fn.__qualname__
            and not hasattr(fn, "__self__")
        ):
            # Check if first parameter is 'self' or 'cls'
            try:
                sig = inspect.signature(fn)
                params = list(sig.parameters.keys())
                if params and params[0] in ("self", "cls"):
                    return "unbound_method"
            except (ValueError, TypeError):
                pass

        return "function"

    # Fallback for other callables
    return "callable"


def call(
    obj: Any, path: DottyPath, args: Optional[List[Any]] = None, kwargs: Optional[dict] = None
) -> Any:
    """
    Call a function or method at the given path with the provided arguments.
    Automatically handles different types of callables including bound methods,
    unbound methods, class constructors, and regular functions.

    For unbound methods, automatically infers the instance (self) from the parent
    object in the path. For example, calling obj.instance.method will automatically
    pass obj.instance as the first argument to the unbound method.

    Args:
      obj: The object to query for the callable
      path: The path to the callable (string, list, or tuple)
      args: List of positional arguments to pass to the callable
      kwargs: Dictionary of keyword arguments to pass to the callable

    Returns:
      The result of calling the function/method

    Raises:
      ValueError: If the path doesn't resolve to a callable
      TypeError: If an unbound method is called without an instance and no parent can be inferred
    """
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    maybe_fn = get(obj, path)

    if not callable(maybe_fn):
        raise ValueError(f"Path '{path}' does not resolve to a callable")

    callable_type = _detect_callable_type(maybe_fn)

    # Call based on detected type
    if callable_type == "bound_method":
        return maybe_fn(*args, **kwargs)
    elif callable_type == "unbound_method":
        # Auto-infer self for unbound methods by using parent object in path
        parts = parse_path(path)
        if len(parts) > 1:
            # Get the parent object (everything except the last part of the path)
            parent_path = parts[:-1]
            parent_obj = get(obj, parent_path)
            # Only use parent as self if it's an instance (not a class)
            if parent_obj is not None and not inspect.isclass(parent_obj):
                # Prepend parent object as self to the args
                args = [parent_obj] + list(args)

        if not args:
            raise TypeError("Unbound method requires instance as first argument")
        return maybe_fn(*args, **kwargs)
    elif callable_type == "class_constructor":
        return maybe_fn(*args, **kwargs)
    elif callable_type == "function":
        return maybe_fn(*args, **kwargs)
    else:
        return maybe_fn(*args, **kwargs)


def get(obj: Any, path: DottyPath, default: Any = None) -> Any:
    """
    Gets the value at path of object. If the resolved value is undefined,
    the default value is returned. Works with nested dictionaries, lists,
    class attributes, and properties.

    Args:
        obj: The object to query
        path: The path of the property to get (string, list, or tuple)
        default: The value returned for undefined resolved values. Defaults to None.

    Returns:
        The resolved value, or default if the path doesn't exist.
        Use _MISSING as default to distinguish "path not found" from "value is None".
    """
    if obj is None:
        return default

    parts = parse_path(path)

    try:
        result = obj
        for part in parts:
            if result is None:
                return default

            if isinstance(result, dict) and part in result:
                result = result[part]
            elif (
                isinstance(result, (list, tuple))
                and isinstance(part, int)
                and -len(result) <= part < len(result)
            ):
                result = result[part]
            elif hasattr(result, part):
                result = getattr(result, part)
            else:
                try:
                    # Try to access as dictionary/list even if not isinstance
                    result = result[part]
                except (KeyError, TypeError, IndexError):
                    return default

        return result
    except (KeyError, AttributeError, IndexError, TypeError):
        return default


def set(obj: Any, path: DottyPath, value: Any) -> Any:
    """
    Sets the value at path of object. If a portion of path doesn't exist, it's created.
    Works with nested dictionaries, lists, and class attributes.

    Args:
        obj: The object to modify
        path: The path of the property to set (string, list, or tuple)
        value: The value to set

    Returns:
        The modified object
    """
    if obj is None:
        # If the root is None, create a new dict as the root
        obj = {}

    parts = parse_path(path)
    if not parts:
        return value

    root = obj
    for i, part in enumerate(parts[:-1]):
        last_part = i == len(parts) - 2
        next_part = parts[i + 1]

        # Handle dictionary access
        if isinstance(root, dict):
            if part not in root:
                root[part] = [] if isinstance(next_part, int) else {}
            root = root[part]
        # Handle list access
        elif isinstance(root, list) and isinstance(part, int):
            # Extend list if needed
            while len(root) <= part:
                root.append(None)
            if root[part] is None:
                root[part] = [] if isinstance(next_part, int) else {}
            root = root[part]
        # Handle object attributes
        elif hasattr(root, part):
            current = getattr(root, part)
            if current is None and last_part:
                setattr(root, part, [] if isinstance(next_part, int) else {})
            root = getattr(root, part)
        # Try to set attribute if it doesn't exist
        else:
            try:
                setattr(root, part, [] if isinstance(next_part, int) else {})  # type: ignore[arg-type]
                root = getattr(root, part)  # type: ignore[arg-type]
            except (AttributeError, TypeError):
                # If we can't set the attribute, fall back to treating as dict
                try:
                    root[part] = [] if isinstance(next_part, int) else {}
                    root = root[part]
                except (TypeError, KeyError):
                    # If all else fails, we can't navigate further
                    return obj

    # Set the final value
    last_part = parts[-1]
    if isinstance(root, dict):
        root[last_part] = value
    elif isinstance(root, list) and isinstance(last_part, int):
        # Extend list if needed
        while len(root) <= last_part:
            root.append(None)
        root[last_part] = value
    elif hasattr(root, last_part):
        setattr(root, last_part, value)
    else:
        try:
            # Try to set as attribute
            setattr(root, last_part, value)
        except (AttributeError, TypeError):
            # Fall back to treating as dict/list
            try:
                root[last_part] = value
            except (TypeError, IndexError):
                pass

    return obj


def has(obj: Any, path: DottyPath) -> bool:
    """
    Checks if path is a direct property of object.
    Works with nested dictionaries, lists, and class attributes.

    Args:
        obj: The object to query
        path: The path to check (string, list, or tuple)

    Returns:
        True if path exists, False otherwise
    """
    if obj is None:
        return False

    parts = parse_path(path)
    if not parts:
        return False

    try:
        result = obj
        for part in parts:
            if result is None:
                return False

            if isinstance(result, dict):
                if part not in result:
                    return False
                result = result[part]
            elif isinstance(result, (list, tuple)):
                if not isinstance(part, int) or part < -len(result) or part >= len(result):
                    return False
                result = result[part]
            elif hasattr(result, part):
                result = getattr(result, part)
            else:
                try:
                    # Try to access as dictionary/list even if not isinstance
                    result = result[part]
                except (KeyError, TypeError, IndexError):
                    return False
        return True
    except (KeyError, AttributeError, IndexError, TypeError):
        return False


def update(obj: Any, path: DottyPath, value: Any) -> Any:
    """
    Updates the dictionaty at the path of object. If the leaf value is not
    a dictionary - an error is raised. If a portion of path doesn't exist,
    it's created.

    Args:
        obj: The object to modify
        path: The path of the property to update (string, list, or tuple)
        value: The value to update
    Returns:
        The modified object
    """

    existing = get(obj, path, {})
    if not isinstance(existing, dict):
        raise TypeError(f"Cannot update non-dictionary value at path: {path}")

    existing.update(value)
    set(obj, path, existing)
    return obj


def interpolate(string: str, context: dict) -> str:
    """
    Interpolates the string using the context dictionary.
    Supports dot notation for nested values.

    Args:
        string: The string to interpolate
        context: The context dictionary for interpolation

    Returns:
        The interpolated string
    """
    pattern = re.compile(r"\{([^\}]+)\}")

    def replacer(match):
        path = match.group(1)
        return str(get(context, path, ""))

    return pattern.sub(replacer, string)


def prefix(obj: Any, path: DottyPath, prefix: str) -> Any:
    set(obj, path, prefix + get(obj, path, ""))
    return obj


def pop(obj: Any, path: DottyPath, default: Any = None) -> Any:
    """
    Pops the value at path of object. If the resolved value is undefined,
    the default value is returned. Works with nested dictionaries, lists,
    class attributes, and properties.

    Args:
        obj: The object to query
        path: The path of the property to pop (string, list, or tuple)
        default: The value returned for undefined resolved values

    Returns:
        The popped value or the default value
    """
    value = get(obj, path, default)
    set(obj, path, None)
    return value


def increment(obj: Any, path: DottyPath, amount: int = 1) -> Any:
    current = get(obj, path, 0)
    if not isinstance(current, (int, float)):
        current = 0
    set(obj, path, current + amount)
    return obj


def transform(obj: Any, path: DottyPath, fn) -> Any:
    current = get(obj, path)
    new_value = fn(current)
    set(obj, path, new_value)
    return new_value


def delete(obj: Any, path: DottyPath) -> Any:
    """
    Deletes the value at path of object.
    Works with nested dictionaries, lists, and class attributes.

    Args:
      obj: The object to modify
      path: The path of the property to delete (string, list, or tuple)
    """

    parts = parse_path(path)
    if not parts:
        return obj

    # Resolve parent using existing get() logic to keep behavior consistent
    parent_path = parts[:-1]
    root = get(obj, parent_path)
    if root is None:
        return obj

    last_part = parts[-1]
    if isinstance(root, dict) and last_part in root:
        del root[last_part]
    elif (
        isinstance(root, list)
        and isinstance(last_part, int)
        and -len(root) <= last_part < len(root)
    ):
        del root[last_part]
    elif hasattr(root, last_part):  # type: ignore[arg-type]
        delattr(root, last_part)  # type: ignore[arg-type]
    else:
        try:
            del root[last_part]
        except (KeyError, TypeError, IndexError):
            pass

    return obj
