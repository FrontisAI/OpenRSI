# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

def get_client(client_cfg):
    match client_cfg.api:
        case "openai":
            from dojo.core.solvers.llm_helpers.backends.open_ai import OpenAIClient

            return OpenAIClient(client_cfg)
        case "litellm":
            from dojo.core.solvers.llm_helpers.backends.lite_llm import LiteLLMClient

            return LiteLLMClient(client_cfg)
        case "gdm":
            from dojo.core.solvers.llm_helpers.backends.gdm import GDMClient

            return GDMClient(client_cfg)
        case _:
            raise Exception(f"Unknown API: {client_cfg['api']}")
