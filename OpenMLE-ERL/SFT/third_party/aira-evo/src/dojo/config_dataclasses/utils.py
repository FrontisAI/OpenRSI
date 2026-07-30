# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import is_dataclass, fields
from typing import Any, Type


def dataclass_from_dict(dataclass_type: Type[Any], data: dict) -> Any:
    """Recursively convert a dictionary to a dataclass instance."""
    if not is_dataclass(dataclass_type):
        raise ValueError(f"{dataclass_type} is not a dataclass type")
    init_args = {}
    for field in fields(dataclass_type):
        field_value = data.get(field.name)
        if is_dataclass(field.type) and isinstance(field_value, dict):
            # Recursively convert nested dataclass fields
            init_args[field.name] = dataclass_from_dict(field.type, field_value)
        else:
            init_args[field.name] = field_value
    return dataclass_type(**init_args)
