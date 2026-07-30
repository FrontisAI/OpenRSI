# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf

from dojo.core.solvers.llm_helpers.backends.utils import get_client
from dojo.utils.logger import get_logger
from dojo.core.solvers.llm_helpers.prompt_template import JinjaPrompt

from dojo.config_dataclasses.operators.base import OperatorConfig
from tts_search.airaevo_concurrency import acquire_llm_slot

import logging

log = logging.getLogger(__name__)

logging.getLogger("openai").setLevel(logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.DEBUG)


class GenericLLM:
    def __init__(self, cfg: OperatorConfig) -> None:
        """
        Initializes the GenericLLM with the provided configuration.

        Args:
            cfg: Arbitrary keyword arguments representing the configuration.
        """
        # Convert the configuration dictionary into an OmegaConf object for easier access
        self.cfg = cfg

        # Initialize the LLM client using the provided client configuration
        self.client = get_client(self.cfg.llm.client)

        # LLM Generation Arguments
        generation_kwargs = self.cfg.llm.generation_kwargs or {}
        if OmegaConf.is_config(generation_kwargs):
            generation_kwargs = OmegaConf.to_container(generation_kwargs, resolve=True)
        self.generation_kwargs = dict(generation_kwargs)

        # Set up the prompt templates based on the configuration
        self._set_up_prompts()

        self.logger = get_logger()

        self.call_tracker = 0

    @property
    def client_content_key(self):
        return self.client.client_content_key

    def _set_up_prompts(self) -> None:
        """
        Sets up the prompt templates by instantiating them using Hydra.
        This method checks for the presence of various prompt templates in the configuration
        and instantiates them accordingly.
        """
        # Check and instantiate the system message prompt template if it exists
        self.system_message_prompt_template = JinjaPrompt(self.cfg.system_message_prompt_template)
        self.init_user_message_prompt_template = JinjaPrompt(self.cfg.init_user_message_prompt_template)
        self.user_message_prompt_template = JinjaPrompt(self.cfg.user_message_prompt_template)

    def _query_client_with_limit(
        self,
        messages: List[Dict[str, str]],
        *,
        json_schema: Optional[str] = None,
        function_name: Optional[str] = None,
        function_description: Optional[str] = None,
    ) -> Any:
        with acquire_llm_slot():
            return self.client.query(
                messages,
                json_schema=json_schema,
                function_name=function_name,
                function_description=function_description,
                **self.generation_kwargs,
            )

    def __call__(
        self,
        query_data: Optional[Dict[str, Any]] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        no_user_message: bool = False,
        json_schema: Optional[str] = None,
        function_name: Optional[str] = None,
        function_description: Optional[str] = None,
    ) -> Any:
        """
        Makes a call to the LLM client with the provided query data or messages.

        Args:
            query_data: A dictionary containing data for formatting prompts.
            messages: A list of message dictionaries to send to the LLM.
            no_user_message: If True, does not append a user message to the prompts.
            json_schema: An optional JSON schema string.

        Returns:
            The response from the LLM client.

        Raises:
            AssertionError: If both query_data and messages are None.
        """
        log.warning("sending query to llm")
        self.call_tracker += 1

        # Ensure that at least one of query_data or messages is provided
        assert not (query_data is None and messages is None), (
            "Neither the query_data nor the messages object were specified."
        )

        # If query_data is not provided, directly query the client with the provided messages
        if query_data is None:
            output, usage_stats = self._query_client_with_limit(
                messages,
                json_schema=json_schema,
                function_name=function_name,
                function_description=function_description,
            )
            usage_stats["cumulative_num_llm_calls"] = self.call_tracker
            return output, {"usage": usage_stats, "prompt_messages": messages, "completion_text": str(output)}

        # If messages are not provided, initialize them with a system message using the query_data
        if messages is None:
            messages = [
                {
                    "role": "system",
                    self.client.client_content_key: self.system_message_prompt_template.format(**query_data),
                }
            ]

        # If no_user_message is True, directly query the client without adding a user message
        if no_user_message:
            output, usage_stats = self._query_client_with_limit(
                messages,
                json_schema=json_schema,
                function_name=function_name,
                function_description=function_description,
            )
            usage_stats["cumulative_num_llm_calls"] = self.call_tracker

            return output, {"usage": usage_stats, "prompt_messages": messages, "completion_text": str(output)}

        # Append a user message based on whether it's the first message or a subsequent one
        if len(messages) == 1:
            # First user message uses the initial user message prompt template
            user_message = {
                "role": "user",
                self.client.client_content_key: self.init_user_message_prompt_template.format(**query_data),
            }
        else:
            # Subsequent user messages use the standard user message prompt template
            user_message = {
                "role": "user",
                self.client.client_content_key: self.user_message_prompt_template.format(**query_data),
            }
        messages.append(user_message)

        # Query the client with the updated messages
        output, usage_stats = self._query_client_with_limit(
            messages,
            json_schema=json_schema,
            function_name=function_name,
            function_description=function_description,
        )
        usage_stats["cumulative_num_llm_calls"] = self.call_tracker

        log.warning("got response from llm")
        return output, {"usage": usage_stats, "prompt_messages": messages, "completion_text": str(output)}
