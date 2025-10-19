#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

__author__ = "bibow"

import base64
import json
import logging
import threading
import traceback
import uuid
from decimal import Decimal
from enum import Enum
from io import BytesIO
from queue import Queue
from typing import Any, Dict, List, Optional

import anthropic
import httpx
import pendulum
from httpx import Response

from ai_agent_handler import AIAgentEventHandler
from silvaengine_utility import Utility, convert_decimal_to_number


# ----------------------------
# HTTP/2 Client Configuration
# ----------------------------
class HTTP2Client:
    """
    Singleton HTTP/2 client for enhanced performance.
    Provides connection pooling, multiplexing, and header compression via HTTP/2.
    """

    _instance = None
    _client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HTTP2Client, cls).__new__(cls)
            cls._instance._initialize_client()
        return cls._instance

    def _initialize_client(self):
        """Initialize HTTP/2 client with optimized settings."""
        self._client = httpx.Client(
            http2=True,  # Enable HTTP/2
            limits=httpx.Limits(
                max_connections=100,  # Maximum concurrent connections
                max_keepalive_connections=20,  # Keep connections alive for reuse
                keepalive_expiry=30.0,  # Keep connections alive for 30 seconds
            ),
            timeout=httpx.Timeout(
                connect=10.0,  # Connection timeout
                read=60.0,  # Read timeout
                write=60.0,  # Write timeout
                pool=5.0,  # Pool timeout
            ),
        )

    def get(self, url: str, **kwargs) -> Response:
        """
        Perform HTTP GET request using HTTP/2 client.

        Args:
            url: URL to fetch
            **kwargs: Additional arguments to pass to httpx.get()

        Returns:
            Response object from httpx
        """
        return self._client.get(url, **kwargs)

    def close(self):
        """Close the HTTP/2 client and free resources."""
        if self._client:
            self._client.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


# Initialize global HTTP/2 client instance
http2_client = HTTP2Client()


class AnthropicBetaVersion(str, Enum):
    """Enum for Anthropic API beta versions"""

    MCP_CLIENT = "mcp-client-2025-04-04"
    CODE_EXECUTION = "code-execution-2025-08-25"
    FILES_API = "files-api-2025-04-14"


# ----------------------------
# Anthropic Response Streaming with Function Handling and History
# ----------------------------
class AnthropicEventHandler(AIAgentEventHandler):
    """
    A handler class for managing interactions with the Anthropic API.
    Provides functionality for streaming responses, handling function calls,
    maintaining conversation history, and managing outputs.
    """

    def __init__(
        self,
        logger: logging.Logger,
        agent: Dict[str, Any],
        **setting: Dict[str, Any],
    ) -> None:
        """
        Initializes the Anthropic event handler with required configuration.
        Sets up logging, API client, and model settings.

        Args:
            logger: Logger instance for recording events and errors
            agent: Dictionary containing agent configuration and available tools
            setting: Additional configuration settings as key-value pairs
        """
        AIAgentEventHandler.__init__(self, logger, agent, **setting)

        if all(
            setting.get(k) for k in ["aws_access_key", "aws_secret_key", "aws_region"]
        ):
            aws_credentials = {
                "aws_access_key": setting["aws_access_key"],
                "aws_secret_key": setting["aws_secret_key"],
                "aws_region": setting["aws_region"],
            }
            self.client = anthropic.AnthropicBedrock(**aws_credentials)
        elif all(setting.get(k) for k in ["project_id", "region"]):
            vertex_credentials = {
                "project_id": setting["project_id"],
                "region": setting["region"],
            }
            self.client = anthropic.AnthropicVertex(**vertex_credentials)
        else:
            self.client = anthropic.Anthropic(
                api_key=self.agent["configuration"].get("api_key")
            )

        # Convert Decimal to appropriate types and build model settings (performance optimization)
        self.model_setting = dict(
            {
                k: (
                    int(v)
                    if k == "max_tokens"
                    else float(v) if isinstance(v, Decimal) else v
                )
                for k, v in self.agent["configuration"].items()
                if k not in ["api_key", "text"]
            },
            **{"system": [{"type": "text", "text": self.agent["instructions"]}]},
        )

        # Cache frequently accessed configuration values (performance optimization)
        self.output_format_type = (
            self.model_setting.get("text", {"format": {"type": "text"}})
            .get("format", {"type": "text"})
            .get("type", "text")
        )

        self.assistant_messages = []

        # Enable/disable timeline logging (default: enabled for backward compatibility)
        self.enable_timeline_log = setting.get("enable_timeline_log", False)

        # Initialize timeline tracking
        self._global_start_time = None
        self._ask_model_depth = 0

    def _get_elapsed_time(self) -> float:
        """
        Get elapsed time in milliseconds from the first ask_model call.

        Returns:
            Elapsed time in milliseconds, or 0 if global start time not set
        """
        if not hasattr(self, "_global_start_time") or self._global_start_time is None:
            return 0.0
        return (pendulum.now("UTC") - self._global_start_time).total_seconds() * 1000

    def reset_timeline(self) -> None:
        """
        Reset the global timeline for a new run.
        Should be called at the start of each new user interaction/run.
        """
        self._global_start_time = None
        if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
            self.logger.info(f"[TIMELINE] Timeline reset for new run")

    def invoke_model(self, **kwargs: Dict[str, Any]) -> Any:
        """
        Makes an API call to the Anthropic model with provided messages.
        Handles both streaming and non-streaming requests.

        Args:
            kwargs: Dictionary containing input messages and streaming configuration

        Returns:
            Response from the Anthropic API

        Raises:
            Exception: If the API call fails for any reason
        """
        try:
            invoke_start = pendulum.now("UTC")

            messages = list(
                filter(lambda x: bool(x["content"]) == True, kwargs["input"])
            )
            # Convert any Decimal values to numbers for JSON serialization
            messages = convert_decimal_to_number(messages)

            betas = self._get_betas(messages)
            if betas:
                result = self.client.beta.messages.create(
                    **dict(
                        self.model_setting,
                        **{
                            "messages": messages,
                            "stream": kwargs["stream"],
                            "betas": betas,
                        },
                    )
                )
            else:
                result = self.client.messages.create(
                    **dict(
                        self.model_setting,
                        **{"messages": messages, "stream": kwargs["stream"]},
                    )
                )

            invoke_end = pendulum.now("UTC")
            invoke_time = (invoke_end - invoke_start).total_seconds() * 1000
            if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                elapsed = self._get_elapsed_time()
                self.logger.info(
                    f"[TIMELINE] T+{elapsed:.2f}ms: API call returned (took {invoke_time:.2f}ms)"
                )

            return result
        except Exception as e:
            self.logger.error(f"Error invoking model: {str(e)}")
            raise Exception(f"Failed to invoke model: {str(e)}")

    @Utility.performance_monitor.monitor_operation(operation_name="Anthorpic")
    def ask_model(
        self,
        input_messages: List[Dict[str, Any]],
        queue: Queue = None,
        stream_event: threading.Event = None,
        input_files: List[str, Any] = [],
        model_setting: Dict[str, Any] = None,
    ) -> Optional[str]:
        """
        Primary method for sending requests to the Anthropic API.
        Supports both streaming and non-streaming modes, with optional model configuration overrides.

        Args:
            input_messages: List of messages representing conversation history and current query
            queue: Optional queue for handling streaming responses
            stream_event: Optional event for signaling stream completion
            input_files: Optional list of input files to process
            model_setting: Optional dictionary to override default model settings

        Returns:
            String containing response ID for non-streaming requests, None for streaming

        Raises:
            Exception: If request processing fails
        """
        # Track preparation time
        ask_model_start = pendulum.now("UTC")

        # Track recursion depth to identify top-level vs recursive calls
        if not hasattr(self, "_ask_model_depth"):
            self._ask_model_depth = 0

        self._ask_model_depth += 1
        is_top_level = self._ask_model_depth == 1

        # Initialize global start time only on top-level ask_model call
        # Recursive calls will use the same start time for the entire run timeline
        if is_top_level:
            self._global_start_time = ask_model_start
            if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    f"[TIMELINE] T+0ms: Run started - First ask_model call"
                )
        else:
            if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                elapsed = self._get_elapsed_time()
                self.logger.info(
                    f"[TIMELINE] T+{elapsed:.2f}ms: Recursive ask_model call started"
                )

        try:
            if not self.client:
                self.logger.error("No Anthropic client provided.")
                return None

            stream = True if queue is not None else False

            # Add model-specific settings if provided
            if model_setting:
                self.model_setting.update(model_setting)

            # Clean up input messages to remove broken tool sequences (performance optimization)
            cleanup_start = pendulum.now("UTC")
            cleanup_end = pendulum.now("UTC")
            cleanup_time = (cleanup_end - cleanup_start).total_seconds() * 1000

            timestamp = pendulum.now("UTC").int_timestamp
            # Optimized UUID generation - use .hex instead of str() conversion
            run_id = f"run-antropic-{self.model_setting['model']}-{timestamp}-{uuid.uuid4().hex[:8]}"

            if input_files:
                input_messages = self._process_input_files(input_files, input_messages)

            # Track total preparation time before API call
            preparation_end = pendulum.now("UTC")
            preparation_time = (
                preparation_end - ask_model_start
            ).total_seconds() * 1000
            if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                elapsed = self._get_elapsed_time()
                self.logger.info(
                    f"[TIMELINE] T+{elapsed:.2f}ms: Preparation complete (took {preparation_time:.2f}ms, cleanup: {cleanup_time:.2f}ms)"
                )

            response = self.invoke_model(
                **{
                    "input": input_messages,
                    "stream": stream,
                }
            )

            # If streaming is enabled, process chunks
            if stream:
                queue.put({"name": "run_id", "value": run_id})
                self.handle_stream(
                    response,
                    input_messages,
                    stream_event=stream_event,
                )
                return None

            self.handle_response(response, input_messages)
            return run_id

        except Exception as e:
            self.logger.error(f"Error in ask_model: {str(e)}")
            raise Exception(f"Failed to process model request: {str(e)}")
        finally:
            # Decrement depth when exiting ask_model
            self._ask_model_depth -= 1

            # Reset timeline when returning to depth 0 (top-level call complete)
            if self._ask_model_depth == 0:
                if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                    elapsed = self._get_elapsed_time()
                    self.logger.info(
                        f"[TIMELINE] T+{elapsed:.2f}ms: Run complete - Resetting timeline"
                    )
                self._global_start_time = None

    def _get_betas(self, input_messages: List[Dict[str, Any]]) -> List[str]:
        """
        Check if input messages contain file content.

        Args:
            input_messages: List of message dictionaries to check

        Returns:
            bool: True if messages contain file content, False otherwise
        """
        betas = []
        if "mcp_servers" in self.model_setting:
            betas.append(AnthropicBetaVersion.MCP_CLIENT.value)

        if any(
            tool["name"] == "code_execution"
            for tool in self.model_setting.get("tools", [])
        ):
            betas.append(AnthropicBetaVersion.CODE_EXECUTION.value)

        for message in input_messages:
            if isinstance(message.get("content"), list):
                for content in message["content"]:
                    if (
                        content.get("type") == "document"
                        and AnthropicBetaVersion.FILES_API.value not in betas
                    ):
                        betas.append(AnthropicBetaVersion.FILES_API.value)
                    elif (
                        content.get("type") == "container_upload"
                        and AnthropicBetaVersion.CODE_EXECUTION.value not in betas
                    ):
                        betas.append(AnthropicBetaVersion.CODE_EXECUTION.value)
        return betas

    def _process_input_files(
        self, input_files: List[Dict[str, Any]], input_messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process and upload input files, attaching them to either code interpreter or user message.

        Args:
            input_files: List of file dictionaries containing file data
            input_messages: List of conversation messages

        Returns:
            Updated input_messages with file references
        """
        code_execution_tool = next(
            (
                tool
                for tool in self.model_setting.get("tools", [])
                if tool.get("type") == "code_execution_20250522"
            ),
            None,
        )

        # Upload each file to OpenAI and store metadata
        file_ids = []
        for input_file in input_files:
            uploaded_file = self.insert_file(**input_file)
            file_ids.append(uploaded_file.id)
            self.uploaded_files.append(
                {
                    "file_id": uploaded_file.id,
                    "code_execution_tool": True if code_execution_tool else False,
                }
            )

        # If code interpreter not available, attach to user message
        if input_messages and input_messages[-1]["role"] == "user":
            # Construct message content with original text and file references
            message_content = [{"type": "text", "text": input_messages[-1]["content"]}]

            if code_execution_tool:
                message_content.extend(
                    {"type": "container_upload", "file_id": file_id}
                    for file_id in file_ids
                )

            else:
                message_content.extend(
                    {"type": "document", "source": {"type": "file", "file_id": file_id}}
                    for file_id in file_ids
                )

            # Update the last message with combined content
            input_messages[-1]["content"] = message_content

        return input_messages

    def handle_function_call(
        self, tool_call: Any, input_messages: List[Dict[str, Any]]
    ) -> None:
        """
        Comprehensive handler for processing and executing function calls from model responses.
        Manages the entire lifecycle of a function call including setup, execution, and result handling.

        Args:
            tool_call: Object containing function call details from the model
            input_messages: Current conversation history to be updated

        Returns:
            Updated input messages list

        Raises:
            ValueError: For invalid tool calls
            Exception: For function execution failures
        """
        # Track function call timing
        function_call_start = pendulum.now("UTC")

        try:
            # Extract function call metadata
            function_call_data = {
                "id": tool_call["id"],
                "arguments": tool_call["input"],
                "name": tool_call["name"],
                "type": tool_call["type"],
            }

            function_name = function_call_data["name"]

            # Record initial function call
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    f"[handle_function_call] Starting function call recording for {function_name}"
                )
            self._record_function_call_start(function_call_data)

            # Parse and process arguments
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    f"[handle_function_call] Processing arguments for function {function_name}"
                )
            arguments = self._process_function_arguments(function_call_data)

            # Execute function and handle result
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    f"[handle_function_call] Executing function {function_name} with arguments {arguments}"
                )
            function_output = self._execute_function(function_call_data, arguments)

            # Update conversation history
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    f"[handle_function_call][{function_name}] Updating conversation history"
                )
            self._update_conversation_history(
                function_call_data, function_output, input_messages
            )

            if self._run is None:
                self._short_term_memory.append(
                    {
                        "message": {
                            "role": self.agent["tool_call_role"],
                            "content": Utility.json_dumps(
                                {
                                    "tool": {
                                        "tool_call_id": function_call_data["id"],
                                        "tool_type": function_call_data["type"],
                                        "name": function_call_data["name"],
                                        "arguments": arguments,
                                    },
                                    "output": function_output,
                                }
                            ),
                        },
                        "created_at": pendulum.now("UTC"),
                    }
                )

            # Log function call execution time
            function_call_end = pendulum.now("UTC")
            function_call_time = (
                function_call_end - function_call_start
            ).total_seconds() * 1000
            if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                elapsed = self._get_elapsed_time()
                self.logger.info(
                    f"[TIMELINE] T+{elapsed:.2f}ms: Function '{function_call_data['name']}' complete (took {function_call_time:.2f}ms)"
                )

            return input_messages

        except Exception as e:
            self.logger.error(f"Error in handle_function_call: {e}")
            raise

    def _record_function_call_start(self, function_call_data: Dict[str, Any]) -> None:
        """
        Records the initiation of a function call in the async storage system.
        Creates an initial record with basic function metadata.

        Args:
            function_call_data: Dictionary containing function call metadata
        """
        self.invoke_async_funct(
            "async_insert_update_tool_call",
            **{
                "tool_call_id": function_call_data["id"],
                "tool_type": function_call_data["type"],
                "name": function_call_data["name"],
            },
        )

    def _process_function_arguments(
        self, function_call_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Processes and validates function arguments from the model response.
        Adds required metadata and performs error handling.

        Args:
            function_call_data: Raw function call data from model

        Returns:
            Processed and validated arguments dictionary

        Raises:
            ValueError: If argument processing fails
        """
        try:
            arguments = function_call_data.get("arguments", {})

            return arguments

        except Exception as e:
            log = traceback.format_exc()
            self.invoke_async_funct(
                "async_insert_update_tool_call",
                **{
                    "tool_call_id": function_call_data["id"],
                    "arguments": function_call_data.get("arguments", "{}"),
                    "status": "failed",
                    "notes": log,
                },
            )
            if self.logger.isEnabledFor(logging.ERROR):
                self.logger.error("Error parsing function arguments: %s", e)
            raise ValueError(f"Failed to parse function arguments: {e}")

    def _execute_function(
        self, function_call_data: Dict[str, Any], arguments: Dict[str, Any]
    ) -> Any:
        """
        Executes the requested function with provided arguments.
        Handles function lookup, execution, and result processing.

        Args:
            function_call_data: Metadata about the function to execute
            arguments: Processed arguments to pass to the function

        Returns:
            Function execution result or error message

        Raises:
            ValueError: If requested function is not supported
        """
        agent_function = self.get_function(function_call_data["name"])
        if not agent_function:
            raise ValueError(
                f"Unsupported function requested: {function_call_data['name']}"
            )

        try:
            # Cache JSON serialization to avoid duplicate work (performance optimization)
            arguments_json = Utility.json_dumps(arguments)

            self.invoke_async_funct(
                "async_insert_update_tool_call",
                **{
                    "tool_call_id": function_call_data["id"],
                    "arguments": arguments_json,
                    "status": "in_progress",
                },
            )

            # Track actual function execution time
            function_exec_start = pendulum.now("UTC")
            function_output = agent_function(**arguments)
            function_exec_end = pendulum.now("UTC")
            function_exec_time = (
                function_exec_end - function_exec_start
            ).total_seconds() * 1000

            if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                elapsed = self._get_elapsed_time()
                self.logger.info(
                    f"[TIMELINE] T+{elapsed:.2f}ms: Function '{function_call_data['name']}' executed (took {function_exec_time:.2f}ms)"
                )

            self.invoke_async_funct(
                "async_insert_update_tool_call",
                **{
                    "tool_call_id": function_call_data["id"],
                    "content": Utility.json_dumps(function_output),
                    "status": "completed",
                },
            )
            return function_output

        except Exception as e:
            log = traceback.format_exc()
            # Cache JSON serialization to avoid duplicate work (performance optimization)
            arguments_json = Utility.json_dumps(arguments)
            self.invoke_async_funct(
                "async_insert_update_tool_call",
                **{
                    "tool_call_id": function_call_data["id"],
                    "arguments": arguments_json,
                    "status": "failed",
                    "notes": log,
                },
            )
            return f"Function execution failed: {e}"

    def _update_conversation_history(
        self,
        function_call_data: Dict[str, Any],
        function_output: Any,
        input_messages: List[Dict[str, Any]],
    ) -> None:
        """
        Updates the conversation history with function call results.
        Formats and appends function output as a user message.

        Args:
            function_call_data: Metadata about the executed function
            function_output: Result from function execution
            input_messages: Current conversation history to update
        """

        input_messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": function_call_data[
                            "id"
                        ],  # from the API response
                        "content": str(function_output),  # from running your tool
                    }
                ],
            }
        )  # Append the function response

    def handle_response(
        self,
        response: Any,
        input_messages: List[Dict[str, Any]],
    ) -> None:
        """
        Processes a complete response from the model.
        Handles both text responses, tool use cases, and MCP tool calls.

        Args:
            response: Complete response object from the model
            input_messages: Current conversation history to update
        """

        contents = []
        if response.stop_reason == "tool_use":
            for content in response.content:
                if content.type == "text":
                    contents.append({"type": content.type, "text": content.text})
                    self.assistant_messages.append(
                        {
                            "content": content.text,
                        }
                    )

                elif content.type == "tool_use":
                    tool_call = {
                        "type": content.type,
                        "id": content.id,
                        "name": content.name,
                        "input": content.input,
                    }
                    contents.append(
                        {
                            "type": content.type,
                            "id": content.id,
                            "name": content.name,
                            "input": content.input,
                        },
                    )
                    input_messages.append({"role": "assistant", "content": contents})
                    input_messages = self.handle_function_call(
                        tool_call, input_messages
                    )
                    response = self.invoke_model(
                        **{"input": input_messages, "stream": False}
                    )
                    self.handle_response(response, input_messages)
        else:
            assistant_message_content = ""
            while self.assistant_messages:
                assistant_message = self.assistant_messages.pop()
                assistant_message_content += assistant_message["content"]

            final_content = ""
            output_files = []
            for content in response.content:
                # Handle MCP tool use blocks
                if content.type == "mcp_tool_use":
                    if self.logger.isEnabledFor(logging.INFO):
                        self.logger.info(
                            f"MCP tool call received - ID: {content.id}, Name: {content.name}, Server: {content.server_name}, Input: {content.input}."
                        )
                    continue
                    # Note: MCP tools are handled automatically by the API
                    # No need to manually execute them like regular tools
                # Handle MCP tool result blocks
                elif content.type == "mcp_tool_result":
                    if self.logger.isEnabledFor(logging.INFO):
                        self.logger.info(
                            f"MCP tool result received - Tool Use ID: {content.tool_use_id}, Content: {content.content}"
                        )
                    continue
                elif content.type == "code_execution_tool_result":
                    if content.content.type == "code_execution_result":
                        for file in content.content.content:
                            output_files.append({"file_id": file["file_id"]})
                    continue
                elif content.type == "BetaServerToolUseBlock":
                    continue
                elif content.type == "server_tool_use":
                    continue
                elif not hasattr(content, "text"):
                    self.logger.error(
                        f"Unexpected content type in response: {content.type}"
                    )
                    continue
                final_content += content.text

            if assistant_message_content:
                final_content = assistant_message_content + " " + final_content

            self.final_output = {
                "message_id": response.id,
                "role": response.role,
                "content": final_content,
                "output_files": output_files,
            }

    def handle_stream(
        self,
        response_stream: Any,
        input_messages: List[Dict[str, Any]] = None,
        stream_event: threading.Event = None,
    ) -> None:
        """
        Processes streaming responses from the model chunk by chunk.
        Handles text content, tool use, MCP tool calls, and maintains state across chunks.

        Note: MCP tools are processed automatically by the MCP connector and don't require
        manual result handling or recursive calls.

        Args:
            response_stream: Iterator yielding response chunks
            input_messages: Optional conversation history to update
            stream_event: Optional event to signal stream completion
        """
        message_id = None
        json_input_parts = []
        stop_reason = None
        tool_use_data = None
        mcp_tool_use_data = None
        mcp_tool_result_data = None
        output_files = []
        self.accumulated_text = ""
        accumulated_partial_json = ""
        accumulated_partial_text = ""
        # Use cached output format type (performance optimization)
        output_format = self.output_format_type
        index = 0
        if self.assistant_messages:
            index = self.assistant_messages[-1]["index"]
            self.send_data_to_stream(
                index=index,
                data_format=output_format,
                chunk_delta=" ",
            )
            index += 1

        for chunk in response_stream:

            # Handle message start event
            if chunk.type == "message_start":
                message_id = chunk.message.id

            # Handle text content
            elif chunk.type == "content_block_delta" and hasattr(chunk.delta, "text"):
                if not chunk.delta.text:
                    continue

                print(chunk.delta.text, end="", flush=True)
                if output_format in ["json_object", "json_schema"]:
                    accumulated_partial_json += chunk.delta.text
                    index, self.accumulated_text, accumulated_partial_json = (
                        self.process_and_send_json(
                            index,
                            self.accumulated_text,
                            accumulated_partial_json,
                            output_format,
                        )
                    )
                elif mcp_tool_result_data:
                    mcp_tool_result_data["content"] += chunk.delta.text
                else:
                    self.accumulated_text += chunk.delta.text
                    accumulated_partial_text += chunk.delta.text
                    # Check if text contains XML-style tags and update format
                    index, accumulated_partial_text = self.process_text_content(
                        index, accumulated_partial_text, output_format
                    )

            # Handle regular tool use start
            elif (
                chunk.type == "content_block_start"
                and hasattr(chunk.content_block, "type")
                and chunk.content_block.type == "tool_use"
            ):
                tool_use_data = {
                    "id": chunk.content_block.id,
                    "type": chunk.content_block.type,
                    "name": chunk.content_block.name,
                    "input": {},  # Will be populated from JSON deltas
                }

            #! Handle MCP tool use start
            elif (
                chunk.type == "content_block_start"
                and hasattr(chunk.content_block, "type")
                and chunk.content_block.type == "mcp_tool_call"
            ):
                mcp_tool_use_data = {
                    "id": chunk.content_block.id,
                    "server_name": chunk.content_block.server_name,
                    "name": chunk.content_block.name,
                    "input": {},  # Will be populated from JSON deltas
                }

            #! Handle MCP tool result blocks
            elif (
                chunk.type == "content_block_start"
                and hasattr(chunk.content_block, "type")
                and chunk.content_block.type == "mcp_tool_result"
            ):
                mcp_tool_result_data = {
                    "tool_use_id": chunk.content_block.tool_use_id,
                    "is_error": chunk.content_block.is_error,
                    "content": (
                        chunk.content_block.content
                        if hasattr(chunk.content_block, "content")
                        else ""
                    ),
                }
                if chunk.content_block.is_error:
                    if self.logger.isEnabledFor(logging.INFO):
                        self.logger.info(
                            f"MCP tool result received - Tool Use ID: {mcp_tool_result_data['tool_use_id']}, Error: {mcp_tool_result_data['content']}"
                        )
                    raise Exception(
                        f"MCP tool error: {mcp_tool_result_data['content']}"
                    )
            elif (
                chunk.type == "content_block_start"
                and hasattr(chunk.content_block, "type")
                and chunk.content_block.type == "code_execution_tool_result"
            ):
                # Handle code execution results in streaming
                if hasattr(chunk.content_block, "content"):
                    # Look for files in the execution result
                    for file in chunk.content_block.content.contnet:
                        output_files.append({"file_id": file["file_id"]})

            # Handle tool input JSON parts (for both regular and MCP tools)
            elif (
                chunk.type == "content_block_delta"
                and hasattr(chunk.delta, "type")
                and chunk.delta.type == "input_json_delta"
            ):
                json_input_parts.append(chunk.delta.partial_json)

            # Handle message delta for stop reason
            elif chunk.type == "message_delta":
                stop_reason = chunk.delta.stop_reason

            elif chunk.type == "message_stop":
                if len(accumulated_partial_text) > 0:
                    self.send_data_to_stream(
                        index=index,
                        data_format=output_format,
                        chunk_delta=accumulated_partial_text,
                    )
                    accumulated_partial_text = ""
                    index += 1

        # Process JSON input if we have tool use (regular or MCP)
        if (tool_use_data or mcp_tool_use_data) and json_input_parts:
            try:
                # Join JSON parts and parse
                json_str = "".join(json_input_parts).strip()

                # Only parse if we have actual content
                if json_str:
                    parsed_input = Utility.json_loads(json_str)

                    if tool_use_data:
                        tool_use_data["input"] = parsed_input
                    elif mcp_tool_use_data:
                        mcp_tool_use_data["input"] = parsed_input
                else:
                    # Empty JSON string, set empty dict as input
                    if self.logger.isEnabledFor(logging.WARNING):
                        self.logger.warning(
                            "Empty JSON input for tool call, using empty dict"
                        )
                    if tool_use_data:
                        tool_use_data["input"] = {}
                    elif mcp_tool_use_data:
                        mcp_tool_use_data["input"] = {}

            except json.JSONDecodeError as e:
                # Log the actual JSON string that failed to parse for debugging
                if self.logger.isEnabledFor(logging.ERROR):
                    self.logger.error(
                        f"Error parsing tool input JSON. JSON string: '{json_str}', Error: {e}"
                    )
                raise Exception(f"Error parsing tool input JSON: {e}")

        # Handle regular tool usage
        if stop_reason == "tool_use" and tool_use_data:
            if self.accumulated_text:
                content = [
                    {"type": "text", "text": self.accumulated_text},
                    tool_use_data,
                ]

                self.assistant_messages.append(
                    {
                        "content": self.accumulated_text,
                        "index": index,
                    }
                )
            else:
                content = [tool_use_data]
            input_messages.append(
                {
                    "role": "assistant",
                    "content": content,
                }
            )

            input_messages = self.handle_function_call(tool_use_data, input_messages)
            response = self.invoke_model(
                **{
                    "input": input_messages,
                    "stream": bool(stream_event),
                }
            )
            self.handle_stream(
                response, input_messages=input_messages, stream_event=stream_event
            )
            return

        # Handle MCP tool usage - MCP tools are processed automatically by the connector
        # We just need to track them for logging/display purposes
        elif mcp_tool_use_data:
            # MCP tools are handled automatically, no recursive call needed
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    f"MCP tool '{mcp_tool_use_data.get('name')}' from server '{mcp_tool_use_data.get('server_name')}' was executed"
                )
        elif mcp_tool_result_data:
            # MCP tool results are handled automatically, no need to process them here
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    f"MCP tool result received - Tool Use ID: {mcp_tool_result_data.get('tool_use_id')}, Content: {mcp_tool_result_data.get('content')}"
                )

        self.send_data_to_stream(
            index=index,
            data_format=output_format,
            is_message_end=True,
        )

        assistant_message_content = ""
        while self.assistant_messages:
            assistant_message = self.assistant_messages.pop()
            assistant_message_content += assistant_message["content"]

        if assistant_message_content:
            self.accumulated_text = (
                assistant_message_content + " " + self.accumulated_text
            )

        self.final_output = {
            "message_id": message_id,
            "role": "assistant",
            "content": self.accumulated_text,
            "output_files": output_files,
        }

        # Signal that streaming has finished
        if stream_event:
            stream_event.set()

    def insert_file(self, **kwargs: Dict[str, Any]) -> Any:
        if "encoded_content" in kwargs:
            encoded_content = kwargs["encoded_content"]
            # Decode the Base64 string
            decoded_content = base64.b64decode(encoded_content)

            # Save the decoded content into a BytesIO object
            content_io = BytesIO(decoded_content)

            # Assign a filename to the BytesIO object
            content_io.name = kwargs["filename"]
        elif "file_uri" in kwargs:
            # Use HTTP/2 client for enhanced performance
            content_io = BytesIO(http2_client.get(kwargs["file_uri"]).content)
            content_io.name = kwargs["filename"]
        else:
            raise Exception("No file content provided")

        file = self.client.beta.files.upload(
            file=(kwargs["filename"], content_io, kwargs["mime_type"]),
        )
        return file

    def get_file(self, **kwargs: Dict[str, Any]) -> Dict[str, Any]:

        file = self.client.beta.files.retrieve_metadata(kwargs["file_id"])
        uploaded_file = {
            "id": file.id,
            "type": file.type,
            "filename": file.filename,
            "size_bytes": file.size_bytes,
            "created_at": pendulum.from_timestamp(file.created_at, tz="UTC"),
            "mime_type": file.mime_type,
            "downloadable": file.downloadable,
        }
        if (
            "encoded_content" in kwargs
            and kwargs["encoded_content"] == True
            and file.downloadable
        ):
            response: Response = self.client.beta.files.download(kwargs["file_id"])
            content = response.content  # Get the actual bytes data)
            # Convert the content to a Base64-encoded string
            uploaded_file["encoded_content"] = base64.b64encode(content).decode("utf-8")

        return uploaded_file
