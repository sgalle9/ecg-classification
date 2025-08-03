import inspect
import pkgutil
from typing import Any

from sklearn.pipeline import Pipeline


def get_preprocessor(transforms_config: dict[str, Any]):
    """Constructs preprocessor pipeline from a configuration dictionary."""

    if transforms_config is None:
        return None

    transform_list = []
    for name, params in transforms_config.items():
        transform_obj = pkgutil.resolve_name(name)

        # Resolve special string arguments.
        params = params.copy()
        for key, value in params.items():
            if isinstance(value, str) and "." in value:
                params[key] = pkgutil.resolve_name(value)

        # if it's a class, instantiate it.
        if inspect.isclass(transform_obj):
            transform_obj = transform_obj(**params)

        transform_name = name.rsplit(".")[-1]  # Remove path.
        transform_list.append((transform_name, transform_obj))

    return Pipeline(steps=transform_list)
