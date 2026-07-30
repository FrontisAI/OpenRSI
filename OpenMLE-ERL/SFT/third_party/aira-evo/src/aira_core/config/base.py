# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import inspect
from dataclasses import dataclass, fields
from typing import Any


@dataclass
class BaseConfig:
    """
    BaseConfig is a base class for all configuration classes.
    """

    def configure_for_debug(self) -> None:
        """
        This function recursively calls the `configure_for_debug` function
        for all the attributes (if applicable).
        When implementing this method in the derived classes and overriding
        some fields, you may want to invoke the `configure_for_debug`
        method from the super class.
        ```
        """
        for class_field in fields(self):
            field_value = getattr(self, class_field.name)
            if _check_if_field_value_has_a_callable(field_value=field_value, name_of_callable="configure_for_debug"):
                field_value.configure_for_debug()

    def validate(self) -> None:
        """
        This function recursively calls the `validate` function
        for all the attributes (if applicable).
        """
        for class_field in fields(self):
            field_value = getattr(self, class_field.name)
            if _check_if_field_value_has_a_callable(field_value=field_value, name_of_callable="validate"):
                field_value.validate()

    @classmethod
    def fields_to_exclude_from_hash(cls) -> list[tuple[str, Any]]:
        fields_to_exclude: list[tuple[str, Any]] = []
        for field_name, field_obj in cls.__dataclass_fields__.items():
            field_type = field_obj.type
            # Note: the reason we need to check if the field_type is a class
            # is that some fancy types (e.g. Optional[BaseConfig]) will be instances
            # of a typing object instead of a class. We currently do not support
            # excluding fields with types other than str, int, float from the hash.
            if not (inspect.isclass(field_type) and issubclass(field_type, (str, int, float, BaseConfig))):
                continue

            if issubclass(field_type, BaseConfig):
                fields_to_exclude.extend(
                    (f"{field_name}.{nested_field_name}", nested_field_type)
                    for nested_field_name, nested_field_type in field_type.fields_to_exclude_from_hash()
                )
            elif field_obj.metadata.get("exclude_from_hash"):
                fields_to_exclude.append((field_name, field_type))

        return fields_to_exclude

    def hash(self) -> str:
        """
        This function hashes all fields of the dataclass
        except those explicitly with `exclude_from_hash=True`
        specified in the field metadata
        """
        fields_to_exclude = [field for field, _ in self.fields_to_exclude_from_hash()]
        fields_to_hash = {}
        for class_field in fields(self):
            if class_field.name in fields_to_exclude:
                continue
            field_value = getattr(self, class_field.name)
            if isinstance(field_value, BaseConfig):
                fields_to_hash[class_field.name] = field_value.hash()
            else:
                fields_to_hash[class_field.name] = str(field_value)

        final_string_to_hash = "-".join(sorted(f"{key}_{value}" for key, value in fields_to_hash.items()))
        return hashlib.sha224(final_string_to_hash.encode()).hexdigest()


def _check_if_field_value_has_a_callable(field_value: Any, name_of_callable: str):
    return hasattr(field_value, name_of_callable) and callable(getattr(field_value, name_of_callable))
