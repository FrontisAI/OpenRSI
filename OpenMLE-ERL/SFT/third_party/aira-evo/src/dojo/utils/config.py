# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


def build(cfg, cfg_obj_map, **kwargs):
    return cfg_obj_map[cfg.__class__.__name__](cfg, **kwargs)
