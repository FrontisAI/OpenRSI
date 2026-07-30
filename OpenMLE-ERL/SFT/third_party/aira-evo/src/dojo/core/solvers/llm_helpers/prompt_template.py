# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from jinja2 import Environment, StrictUndefined
import hydra

from dojo.config_dataclasses.llm.jinjaprompt import JinjaPromptConfig


class JinjaPrompt:
    r"""This class can be used to generate prompts from jinja templates

    :param \**kwargs:
        See below:
    :Keyword Arguments:
        * *input_variables* (``List[str]``) --
            A list of variables that are required to render the template
        * *partial_variables* (``Dict[str, Any]``) --
            A dictionary of variables and their values that are required to render the template (useful when one has some variables before others)
        * *template* (``str``) --
            The jinja template to render
    """

    def __init__(self, cfg: JinjaPromptConfig):
        # This is temporary -- under the desired setup this will be done in main under a single call
        cfg = hydra.utils.instantiate(cfg, _recursive_=False)
        self.input_variables: set = set(cfg.input_variables)
        self.partial_variables = cfg.partial_variables
        self.template: str = cfg.template
        self.environment = Environment(undefined=StrictUndefined)

    def format(self, **kwargs):
        r"""format the template with the given input variables

        :param \**kwargs: The input variables to render the template (should be a subset of the input variables)
        :return: The rendered template
        :rtype: str
        """
        template = self.environment.from_string(self.template)
        merged_args = {**self.partial_variables, **kwargs}
        return template.render(**merged_args)
