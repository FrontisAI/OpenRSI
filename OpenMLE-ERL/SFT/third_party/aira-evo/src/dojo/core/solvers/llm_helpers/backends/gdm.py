# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Portions of this file are MIT-licensed
# Copyright (c) 2024 Weco AI Ltd
# See THIRD_PARTY_LICENSES.md for the full licence text.
# https://github.com/WecoAI/aideml/blob/main/LICENSE

import json
import logging
import os
from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from google import genai
from google.genai import types
import jsonschema
from omegaconf import OmegaConf
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


class GDMClient:
    PromptType = Union[str, Dict[str, Any], List[Any]]
    FunctionCallType = Dict[str, Any]
    OutputType = Union[str, FunctionCallType]

    def __init__(self, client_cfg):
        self.model = client_cfg.model_id
        self.api_key = os.getenv("GEMINI_API_KEY", "")

        self._client = genai.Client(api_key=self.api_key)

    @property
    def client_content_key(self):
        return "content"

    def generate_response(self, messages: list, functions: list = None, **model_kwargs):
        """
        Call the Gemini model with a conversation and optional function definitions.
        Returns a tuple: (output_text, usage_stats).
        """
        # Separate system instruction if present
        system_instruction = None
        content_list = []
        for msg in messages:
            role = msg.get("role")
            text = msg.get("content", "")
            if role == "system":
                system_instruction = text  # use config for system instructions
            elif role == "user":
                content_list.append(types.Content(role="user", parts=[types.Part.from_text(text=text)]))
            elif role in ("assistant", "model"):
                # Treat assistant messages as 'model' role in Gemini
                content_list.append(types.Content(role="model", parts=[types.Part.from_text(text=text)]))

        # Prepare function tools if any functions are provided
        tools = None
        if functions:
            func_decls = []
            for func in functions:
                # Convert OpenAI function spec to Gemini FunctionDeclaration
                name = func.get("name")
                desc = func.get("description", "")
                params = func.get("parameters", {})

                # Convert parameters JSON schema to types.Schema recursively
                def to_schema(schema):
                    schema_type = schema.get("type", "")
                    # OpenAI uses lowercase types; Gemini expects uppercase (OBJECT, STRING, etc.)
                    schema_obj = types.Schema(type=schema_type.upper() if schema_type else None)
                    if "description" in schema:
                        schema_obj.description = schema["description"]
                    if "properties" in schema:
                        schema_obj.properties = {key: to_schema(val) for key, val in schema["properties"].items()}
                    if "required" in schema:
                        schema_obj.required = schema["required"]
                    return schema_obj

                param_schema = to_schema(params)
                func_decls.append(types.FunctionDeclaration(name=name, description=desc, parameters=param_schema))
            if func_decls:
                tools = [types.Tool(function_declarations=func_decls)]

        # Map model_kwargs to GenerateContentConfig
        config_args = {}
        # Map known parameter names to Gemini config fields
        if system_instruction:
            config_args["system_instruction"] = system_instruction
        if "temperature" in model_kwargs:
            config_args["temperature"] = model_kwargs["temperature"]
        if "top_p" in model_kwargs:
            config_args["top_p"] = model_kwargs["top_p"]
        if "top_k" in model_kwargs:
            config_args["top_k"] = model_kwargs["top_k"]
        # max_tokens for output
        if "max_tokens" in model_kwargs:
            config_args["max_output_tokens"] = model_kwargs["max_tokens"]
        if "max_output_tokens" in model_kwargs:
            config_args["max_output_tokens"] = model_kwargs["max_output_tokens"]
        if "presence_penalty" in model_kwargs:
            config_args["presence_penalty"] = model_kwargs["presence_penalty"]
        if "frequency_penalty" in model_kwargs:
            config_args["frequency_penalty"] = model_kwargs["frequency_penalty"]
        if "stop" in model_kwargs:
            # `stop` could be a single string or list of strings
            stops = model_kwargs["stop"]
            config_args["stop_sequences"] = stops if isinstance(stops, list) else [stops]
        # Include function calling config if tools are present
        if tools:
            config_args["tool_config"] = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="ANY")
            )
            config_args["tools"] = tools

        # Create config object
        config = types.GenerateContentConfig(**config_args)

        # Call the Gemini model
        # Record start time for latency measurement
        start_time = time.monotonic()
        response = self._client.models.generate_content(
            model=self.model,
            contents=content_list if len(content_list) > 1 else (content_list[0] if content_list else ""),
            # If only one content element, can pass it directly (string or Content)
            config=config,
        )
        # Calculate latency
        latency = time.monotonic() - start_time

        # Parse the response
        if tools is None:
            # No function calling was used
            output = response.text
        else:
            # Attempt to extract function call
            function_call = response.function_calls
            if not function_call:
                logger.warning(
                    "No function call was used despite function spec. Fallback to text.\n"
                    f"Message content: {response.text}"
                )
                output = response.text
            else:
                function_call = function_call[0]
                try:
                    if isinstance(function_call.args, str):
                        output = json.load(function_call.args)
                    elif isinstance(function_call.args, dict):
                        output = function_call.args
                except json.JSONDecodeError as ex:
                    logger.error(f"Error decoding function arguments:\n{str(function_call.args)}")
                    raise ex

        # Gather token usage stats
        usage = response.usage_metadata  # usage stats object
        prompt_tokens = getattr(usage, "prompt_token_count", None)
        # Gemini may provide list of candidate token counts
        completion_tokens = None
        if hasattr(usage, "candidates_token_count") and usage.candidates_token_count:
            # Use first candidate's token count as completion tokens
            completion_tokens = usage.candidates_token_count
        total_tokens = getattr(usage, "total_token_count", None)
        # Fallback: if completion not provided, compute if possible
        if prompt_tokens is not None and total_tokens is not None and completion_tokens is None:
            completion_tokens = total_tokens - prompt_tokens
        usage_stats = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency": latency,
        }
        return output, usage_stats

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
        functions = None
        if func_spec is not None:
            functions = [func_spec.as_openai_tool_dict]

        try:
            output, usage_stats = self.generate_response(messages, functions, **filtered_kwargs)
        except Exception as e:
            # Check if function calling is not supported
            if "function calling" in str(e).lower() or "functions" in str(e).lower():
                logger.warning(
                    "Function calling was attempted but is not supported by this model. "
                    "Falling back to plain text generation."
                )

                # Retry without function calling
                output, usage_stats = self.generate_response(messages=messages, functions=None, **filtered_kwargs)
            else:
                # Re-raise other exceptions
                raise

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

        output, usage_stats = self._query_client(
            messages=messages,
            model_kwargs=model_kwargs,
            json_schema=json_schema,
            function_name=function_name,
            function_description=function_description,
        )

        return output, usage_stats
