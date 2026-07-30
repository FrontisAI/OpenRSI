# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

def __getattr__(name: str):
    if name == "Evolutionary":
        from .evo import Evolutionary

        return Evolutionary
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Evolutionary"]
