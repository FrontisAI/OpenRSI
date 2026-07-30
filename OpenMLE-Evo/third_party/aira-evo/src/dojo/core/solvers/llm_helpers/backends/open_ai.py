# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import jsonschema
import openai

# import weave
from dataclasses_json import DataClassJsonMixin

# Configure logging
logger = logging.getLogger("Backend")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


@dataclass
class FunctionSpec(DataClassJsonMixin):
    name: str
    json_schema: Dict[str, Any]  # JSON schema
    description: str

    def __post_init__(self):
        # Validate the JSON schema
        jsonschema.Draft7Validator.check_schema(self.json_schema)

    @property
    def as_openai_tool_dict(self) -> Dict[str, Any]:
        """Convert to OpenAI's function format."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.json_schema,
        }

    @property
    def openai_tool_choice_dict(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name},
        }


class OpenAIClient:
    PromptType = Union[str, Dict[str, Any], List[Any]]
    FunctionCallType = Dict[str, Any]
    OutputType = Union[str, FunctionCallType]

    def __init__(self, client_cfg):
        """
        Initialize the OpenAI client with any desired default arguments or configuration.
        """
        self.model = client_cfg.model_id
        self.base_url = client_cfg.base_url
        self.api_key = os.getenv("PRIMARY_KEY", "")
        self.use_azure_client = client_cfg.use_azure_client

        if self.use_azure_client:
            self._client = openai.AzureOpenAI(
                max_retries=3, api_key=self.api_key, api_version="2024-10-21", azure_endpoint=self.base_url
            )
        else:
            self._client = openai.OpenAI(max_retries=3, base_url=self.base_url, api_key=self.api_key)

        logging.getLogger("httpx").setLevel(logging.WARNING)

    @property
    def client_content_key(self):
        return "content"

    def _query_client(
        self,
        messages: List[Dict[str, str]],
        model_kwargs: Dict[str, Any] = {},
        json_schema: Optional[str] = None,
        function_name: Optional[str] = None,
        function_description: Optional[str] = None,
    ) -> Tuple[OutputType, float, int, int, Dict[str, Any]]:
        model_kwargs["model"] = self.model
        filtered_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

        func_spec = None
        if json_schema and function_name and function_description:
            func_spec = FunctionSpec(function_name, json.loads(json_schema), function_description)

        # Attach function specifications if provided
        if func_spec is not None:
            filtered_kwargs["functions"] = [func_spec.as_openai_tool_dict]
            filtered_kwargs["function_call"] = {"name": function_name}

        completion = None
        start_time = time.monotonic()
        try:
            completion = self._client.chat.completions.create(messages=messages, **filtered_kwargs)
        except openai.BadRequestError as e:
            # Check if function calling is not supported
            if "function calling" in str(e).lower() or "functions" in str(e).lower():
                logger.warning(
                    "Function calling was attempted but is not supported by this model. "
                    "Falling back to plain text generation."
                )
                # Remove function-calling parameters and retry
                filtered_kwargs.pop("functions", None)
                filtered_kwargs.pop("function_call", None)

                # Retry without function calling
                completion = self._client.chat.completions.create(messages=messages, **filtered_kwargs)
            else:
                # Re-raise other exceptions
                raise

        # Calculate latency
        latency = time.monotonic() - start_time

        choice = completion.choices[0]
        if completion is not None:
            usage_stats = completion.to_dict()["usage"]
        else:
            usage_stats = {}
        usage_stats["latency"] = latency

        # Parse the response
        if func_spec is None or "functions" not in filtered_kwargs:
            # No function calling was used
            output = choice.message.content
        else:
            # Attempt to extract function call
            function_call = choice.message.function_call
            if not function_call:
                logger.warning(
                    "No function call was used despite function spec. Fallback to text.\n"
                    f"Message content: {choice.message.content}"
                )
                output = choice.message.content
            else:
                if not str(function_call.name).strip() == str(func_spec.name).strip():
                    logger.warning(
                        f"Function name mismatch: expected {func_spec.name}, "
                        f"got {function_call.name}. Fallback to text."
                    )
                    output = choice.message.content
                else:
                    try:
                        output = json.loads(function_call.arguments)
                    except json.JSONDecodeError as ex:
                        logger.error(f"Error decoding function arguments:\n{function_call.arguments}")
                        raise ex

        return output, usage_stats

    def query(
        self,
        messages: List[Dict[str, str]],
        json_schema: Optional[str] = None,
        function_name: Optional[str] = None,
        function_description: Optional[str] = None,
        **model_kwargs,
    ) -> OutputType:
        """
        General LLM query for various backends with a single system and user message.
        Supports function calling for some backends.

        Args:
            system_message (PromptType | None): Uncompiled system message.
            user_message (PromptType | None): Uncompiled user message.
            model (str): Identifier for the model to use (e.g., "gpt-4-turbo").
            temperature (float | None, optional): Sampling temperature.
            max_tokens (int | None, optional): Maximum number of tokens to generate.
            func_spec (FunctionSpec | None, optional): Optional FunctionSpec for function calling.
            **model_kwargs: Additional keyword arguments for the model.

        Returns:
            OutputType: A string completion or a dict with function call details.
        """

        if self.model == "o1-preview":
            messages = [{"role": "user", self.client_content_key: m[self.client_content_key]} for m in messages]
            if "temperature" in model_kwargs:
                model_kwargs.pop("temperature")

        output, usage_stats = self._query_client(
            messages=messages,
            model_kwargs=model_kwargs,
            json_schema=json_schema,
            function_name=function_name,
            function_description=function_description,
        )

        return output, usage_stats
