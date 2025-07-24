import inspect
import sys

from sklearn import impute, preprocessing
from sklearn.pipeline import Pipeline


def get_preprocessor(config_transforms):
    if config_transforms is None:
        return None

    transform_list = []
    for transform_name, params in config_transforms.items():
        transform_obj = None

        # search for the transform object (class or function)
        for module in [sys.modules[__name__], impute, preprocessing]:
            if hasattr(module, transform_name):
                transform_obj = getattr(module, transform_name)
                break

        if transform_obj is None:
            raise ValueError(f"Transform '{transform_name}' not found in any of the search paths.")

        # if it's a class, instantiate it
        if inspect.isclass(transform_obj):
            transform_obj = transform_obj(**params)

        transform_list.append((transform_name, transform_obj))

    return Pipeline(steps=transform_list)
