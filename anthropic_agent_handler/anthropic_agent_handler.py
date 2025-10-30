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
    WEB_FETCH = "web-fetch-2025-09-10"


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
        self.model_setting = {
            "system": [{"type": "text", "text": self.agent["instructions"]}]
        }

        for k, v in self.agent["configuration"].items():
            if k not in ["api_key", "text"]:
                if k == "max_tokens":
                    self.model_setting[k] = int(v)
                elif k == "temperature":
                    self.model_setting[k] = float(v)
                elif isinstance(v, Decimal):
                    self.model_setting[k] = float(v)
                else:
                    self.model_setting[k] = convert_decimal_to_number(v)

        # Cache frequently accessed configuration values (performance optimization)
        self.output_format_type = (
            self.model_setting.get("text", {"format": {"type": "text"}})
            .get("format", {"type": "text"})
            .get("type", "text")
        )

        self.assistant_messages = []

        # Initialize timeline tracking
        self._global_start_time = None
        self._ask_model_depth = 0
        self.enable_timeline_log = setting.get("enable_timeline_log", False)

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
            self.logger.info("[TIMELINE] Timeline reset for new run")

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

            messages = list(filter(lambda x: bool(x["content"]), kwargs["input"]))
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

            if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                invoke_end = pendulum.now("UTC")
                invoke_time = (invoke_end - invoke_start).total_seconds() * 1000
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
        input_files: List[str] = [],
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
                self.logger.info("[TIMELINE] T+0ms: Run started - First ask_model call")
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

            if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                # Track total preparation time before API call
                preparation_end = pendulum.now("UTC")
                preparation_time = (
                    preparation_end - ask_model_start
                ).total_seconds() * 1000
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

        if any(
            tool["name"] == "web_fetch" for tool in self.model_setting.get("tools", [])
        ):
            betas.append(AnthropicBetaVersion.WEB_FETCH.value)

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
    ) -> List[Dict[str, Any]]:
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

            if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                # Log function call execution time
                function_call_end = pendulum.now("UTC")
                function_call_time = (
                    function_call_end - function_call_start
                ).total_seconds() * 1000
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

            if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                function_exec_end = pendulum.now("UTC")
                function_exec_time = (
                    function_exec_end - function_exec_start
                ).total_seconds() * 1000
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

    def _log_citation(
        self, citation: Any, idx: int, is_streaming: bool = False
    ) -> None:
        """
        Private helper to log a single citation.
        Handles both dict and object formats for citations.

        Args:
            citation: The citation object to log
            idx: The index of the citation (for display purposes)
            is_streaming: Whether this is from streaming mode
        """
        if not self.logger.isEnabledFor(logging.INFO):
            return

        # Check citation type
        citation_type = None
        if isinstance(citation, dict):
            citation_type = citation.get("type", "N/A")
        elif hasattr(citation, "type"):
            citation_type = citation.type

        # Only log web_search_result_location citations
        if citation_type != "web_search_result_location":
            return

        # Extract citation details - handle both dict and object formats
        if isinstance(citation, dict):
            url = citation.get("url", "N/A")
            title = citation.get("title", "N/A")
            cited_text = citation.get("cited_text", "N/A")
        else:
            url = citation.url if hasattr(citation, "url") else "N/A"
            title = citation.title if hasattr(citation, "title") else "N/A"
            cited_text = citation.cited_text if hasattr(citation, "cited_text") else "N/A"

        # Truncate cited text if needed
        cited_text_preview = (
            cited_text[:100] if isinstance(cited_text, str) and len(cited_text) > 100 else cited_text
        )

        # Log based on mode
        if is_streaming:
            self.logger.info(
                f"Citation received in stream - URL: {url}, "
                f"Title: {title}, "
                f"Cited text: {cited_text_preview}..."
            )
        else:
            self.logger.info(
                f"  Citation {idx}: Title: {title}, "
                f"URL: {url}, "
                f"Cited text: {cited_text_preview}..."
            )

    def _handle_citations(
        self, citations_list: List[Any], text_preview: str = None, is_streaming: bool = False
    ) -> None:
        """
        Private helper to handle citations from text blocks.
        Logs citation information for both streaming and non-streaming modes.

        Args:
            citations_list: List of citation objects
            text_preview: Optional preview of the text containing citations
            is_streaming: Whether this is from streaming mode
        """
        if not self.logger.isEnabledFor(logging.INFO):
            return

        if not citations_list:
            return

        # Log header with citation count
        if is_streaming:
            self.logger.info(
                f"Text block with citations started - Citation count: {len(citations_list)}"
            )
        else:
            log_msg = f"Text with citations - Citation count: {len(citations_list)}"
            if text_preview:
                log_msg += f", Text: {text_preview[:100]}..."
            self.logger.info(log_msg)

        # Log first 3 citations
        for idx, citation in enumerate(citations_list[:3], 1):
            self._log_citation(citation, idx, is_streaming=is_streaming)

    def _handle_citation_delta(self, delta: Any) -> None:
        """
        Private helper to handle citation deltas in streaming mode.
        Logs individual citations as they arrive.

        Args:
            delta: The delta object containing citation information
        """
        if not self.logger.isEnabledFor(logging.INFO):
            return

        # Extract citation from delta
        citation = delta.citation if hasattr(delta, "citation") else None
        if citation:
            # Use _log_citation with streaming mode (idx not used for streaming single citations)
            self._log_citation(citation, idx=0, is_streaming=True)

    def _handle_server_tool_use(
        self, content: Any, tool_input: Dict[str, Any] = None, is_streaming: bool = False
    ) -> None:
        """
        Private helper to handle server_tool_use blocks (web_search and web_fetch).
        Logs tool initiation for both streaming and non-streaming modes.

        Args:
            content: The server_tool_use content block (can be a dict or object)
            tool_input: Optional parsed tool input dictionary (for streaming when input is ready)
            is_streaming: Whether this is from streaming mode
        """
        if not self.logger.isEnabledFor(logging.INFO):
            return

        # Extract tool metadata - handle both dict and object formats
        if isinstance(content, dict):
            tool_id = content.get("id", "N/A")
            tool_name = content.get("name", "N/A")
            content_input = content.get("input", {})
        else:
            tool_id = content.id if hasattr(content, "id") else "N/A"
            tool_name = content.name if hasattr(content, "name") else "N/A"
            content_input = content.input if hasattr(content, "input") else {}

        # Use provided tool_input if available (for streaming), otherwise use content_input
        input_to_use = tool_input if tool_input is not None else content_input

        mode_prefix = "in stream" if is_streaming else ""

        # Log based on tool type
        if tool_name == "web_search":
            query = input_to_use.get("query", "N/A") if isinstance(input_to_use, dict) else "N/A"
            if is_streaming and tool_input is None:
                # Initial log when tool starts (before input is parsed)
                self.logger.info(
                    f"Server tool use started in stream - Name: {tool_name}, ID: {tool_id}"
                )
            else:
                # Log with query details
                log_msg = f"Web search query {mode_prefix}" if is_streaming else "Web search initiated"
                self.logger.info(f"{log_msg} - ID: {tool_id}, Query: '{query}'")
        elif tool_name == "web_fetch":
            url = input_to_use.get("url", "N/A") if isinstance(input_to_use, dict) else "N/A"
            if is_streaming and tool_input is None:
                # Initial log when tool starts (before input is parsed)
                self.logger.info(
                    f"Server tool use started in stream - Name: {tool_name}, ID: {tool_id}"
                )
            else:
                # Log with URL details
                log_msg = f"Web fetch URL {mode_prefix}" if is_streaming else "Web fetch initiated"
                self.logger.info(f"{log_msg} - ID: {tool_id}, URL: '{url}'")
        else:
            # Generic server tool logging
            if is_streaming and tool_input is None:
                self.logger.info(
                    f"Server tool use started in stream - Name: {tool_name}, ID: {tool_id}"
                )
            else:
                self.logger.info(
                    f"Server tool use - Name: {tool_name}, ID: {tool_id}, Input: {input_to_use}"
                )

    def _handle_web_search_tool_result(
        self, content: Any, is_streaming: bool = False
    ) -> None:
        """
        Private helper to handle web_search_tool_result blocks.
        Logs search results and errors for both streaming and non-streaming modes.

        Args:
            content: The web_search_tool_result content block
            is_streaming: Whether this is from streaming mode
        """
        tool_use_id = (
            content.tool_use_id if hasattr(content, "tool_use_id") else "N/A"
        )
        mode_prefix = "in stream" if is_streaming else ""

        # Check for errors first (error is in content object, not array)
        search_content = content.content if hasattr(content, "content") else None

        if not search_content:
            return

        # Check if content is an error object (dict or object)
        is_error = False
        if isinstance(search_content, dict):
            content_type = search_content.get("type", "")
            if content_type == "web_search_tool_result_error":
                is_error = True
                if self.logger.isEnabledFor(logging.ERROR):
                    error_code = search_content.get("error_code", "Unknown error")
                    self.logger.error(
                        f"Web search error {mode_prefix} - Tool Use ID: {tool_use_id}, "
                        f"Error code: {error_code}"
                    )
        elif (
            hasattr(search_content, "type")
            and search_content.type == "web_search_tool_result_error"
        ):
            is_error = True
            if self.logger.isEnabledFor(logging.ERROR):
                error_code = (
                    search_content.error_code
                    if hasattr(search_content, "error_code")
                    else "Unknown error"
                )
                self.logger.error(
                    f"Web search error {mode_prefix} - Tool Use ID: {tool_use_id}, "
                    f"Error code: {error_code}"
                )

        if is_error:
            return

        # If not error, content should be an array of results
        if self.logger.isEnabledFor(logging.INFO):
            if isinstance(search_content, list):
                result_count = len(search_content)

                self.logger.info(
                    f"Web search results received {mode_prefix} - Tool Use ID: {tool_use_id}, "
                    f"Results: {result_count}"
                )

                # Log details of each result from content array
                for idx, result in enumerate(search_content[:5], 1):  # Log first 5
                    # Handle both dict and object formats
                    result_type = (
                        result.get("type")
                        if isinstance(result, dict)
                        else (result.type if hasattr(result, "type") else None)
                    )

                    if result_type == "web_search_result":
                        if isinstance(result, dict):
                            result_detail = {
                                "url": result.get("url", "N/A"),
                                "title": result.get("title", "N/A"),
                                "page_age": result.get("page_age", "N/A"),
                            }
                        else:
                            result_detail = {
                                "url": result.url if hasattr(result, "url") else "N/A",
                                "title": (
                                    result.title if hasattr(result, "title") else "N/A"
                                ),
                                "page_age": (
                                    result.page_age
                                    if hasattr(result, "page_age")
                                    else "N/A"
                                ),
                            }
                        self.logger.info(
                            f"  Result {idx}: Title: {result_detail['title']}, "
                            f"URL: {result_detail['url']}, Page Age: {result_detail['page_age']}"
                        )

    def _handle_web_fetch_tool_result(
        self, content: Any, is_streaming: bool = False
    ) -> None:
        """
        Private helper to handle web_fetch_tool_result blocks.
        Logs fetch results and errors for both streaming and non-streaming modes.

        Args:
            content: The web_fetch_tool_result content block
            is_streaming: Whether this is from streaming mode
        """
        tool_use_id = (
            content.tool_use_id if hasattr(content, "tool_use_id") else "N/A"
        )
        mode_prefix = "in stream" if is_streaming else ""

        # Extract from nested structure: content.content (web_fetch_result)
        fetch_result = content.content if hasattr(content, "content") else None

        if not fetch_result:
            return

        # Check for errors first (error_code is inside fetch_result)
        is_error = False
        if isinstance(fetch_result, dict):
            fetch_type = fetch_result.get("type", "")
            if fetch_type == "web_fetch_tool_result_error":
                is_error = True
                if self.logger.isEnabledFor(logging.ERROR):
                    error_code = fetch_result.get("error_code", "Unknown error")
                    self.logger.error(
                        f"Web fetch error {mode_prefix} - Tool Use ID: {tool_use_id}, "
                        f"Error code: {error_code}"
                    )
        else:
            if (
                hasattr(fetch_result, "type")
                and fetch_result.type == "web_fetch_tool_error"
            ):
                is_error = True
                if self.logger.isEnabledFor(logging.ERROR):
                    error_code = (
                        fetch_result.error_code
                        if hasattr(fetch_result, "error_code")
                        else "Unknown error"
                    )
                    self.logger.error(
                        f"Web fetch error {mode_prefix} - Tool Use ID: {tool_use_id}, "
                        f"Error code: {error_code}"
                    )

        if is_error:
            return

        # Handle both dict and object formats for successful fetch
        if self.logger.isEnabledFor(logging.INFO):
            if isinstance(fetch_result, dict):
                url = fetch_result.get("url", "N/A")
                retrieved_at = fetch_result.get("retrieved_at", "N/A")
                result_content = fetch_result.get("content", {})

                # Extract from content.source
                if isinstance(result_content, dict):
                    source = result_content.get("source", {})
                    media_type = (
                        source.get("media_type", "N/A")
                        if isinstance(source, dict)
                        else "N/A"
                    )
                    data = (
                        source.get("data", "N/A")
                        if isinstance(source, dict)
                        else "N/A"
                    )
                    title = result_content.get("title", "N/A")
                else:
                    media_type = "N/A"
                    data = "N/A"
                    title = "N/A"
            else:
                url = fetch_result.url if hasattr(fetch_result, "url") else "N/A"
                retrieved_at = (
                    fetch_result.retrieved_at
                    if hasattr(fetch_result, "retrieved_at")
                    else "N/A"
                )

                # Extract from content.source
                result_content = (
                    fetch_result.content if hasattr(fetch_result, "content") else None
                )
                if result_content:
                    source = (
                        result_content.source
                        if hasattr(result_content, "source")
                        else None
                    )
                    media_type = (
                        source.media_type
                        if source and hasattr(source, "media_type")
                        else "N/A"
                    )
                    data = (
                        source.data if source and hasattr(source, "data") else "N/A"
                    )
                    title = (
                        result_content.title
                        if hasattr(result_content, "title")
                        else "N/A"
                    )
                else:
                    media_type = "N/A"
                    data = "N/A"
                    title = "N/A"

            content_preview = str(data)[:200] if data != "N/A" else "N/A"

            self.logger.info(
                f"Web fetch result received {mode_prefix} - Tool Use ID: {tool_use_id}, "
                f"URL: {url}, "
                f"Retrieved at: {retrieved_at}, "
                f"Title: {title}, "
                f"Media type: {media_type}, "
                f"Content preview: {content_preview}..."
            )

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
                    # Log server tool use (web_search and web_fetch)
                    # In non-streaming mode, input is already available in content.input
                    tool_input = content.input if hasattr(content, "input") else (content.get("input") if isinstance(content, dict) else {})
                    self._handle_server_tool_use(content, tool_input=tool_input, is_streaming=False)
                    continue
                elif content.type == "web_search_tool_result":
                    self._handle_web_search_tool_result(content, is_streaming=False)
                    continue
                elif content.type == "web_fetch_tool_result":
                    self._handle_web_fetch_tool_result(content, is_streaming=False)
                    continue
                elif not hasattr(content, "text"):
                    self.logger.error(
                        f"Unexpected content type in response: {content.type}"
                    )
                    continue

                # Log citations if present in text blocks
                if hasattr(content, "citations") and content.citations:
                    self._handle_citations(
                        content.citations,
                        text_preview=content.text,
                        is_streaming=False
                    )

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
        server_tool_use_data = None
        output_files = []
        self.accumulated_text = ""
        accumulated_partial_json = ""
        accumulated_partial_text = ""
        # Use cached output format type (performance optimization)
        output_format = self.output_format_type

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                f"[handle_stream] Initialized stream state - "
                f"accumulated_partial_json: '{accumulated_partial_json}', "
                f"output_format: '{output_format}'"
            )

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

            # Handle content block start for text (to capture citations metadata)
            elif (
                chunk.type == "content_block_start"
                and hasattr(chunk.content_block, "type")
                and chunk.content_block.type == "text"
            ):
                # Check if this text block has citations
                if (
                    hasattr(chunk.content_block, "citations")
                    and chunk.content_block.citations
                ):
                    self._handle_citations(
                        chunk.content_block.citations,
                        text_preview=None,
                        is_streaming=True
                    )

            # Handle citations delta
            elif (
                chunk.type == "content_block_delta"
                and hasattr(chunk.delta, "type")
                and chunk.delta.type == "citations_delta"
            ):
                self._handle_citation_delta(chunk.delta)

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

            # Handle server tool use (web_search)
            elif (
                chunk.type == "content_block_start"
                and hasattr(chunk.content_block, "type")
                and chunk.content_block.type == "server_tool_use"
            ):
                server_tool_use_data = {
                    "id": chunk.content_block.id,
                    "name": (
                        chunk.content_block.name
                        if hasattr(chunk.content_block, "name")
                        else "N/A"
                    ),
                    "input": {},  # Will be populated from JSON deltas
                }
                # Log initial server tool use (before input is parsed)
                self._handle_server_tool_use(
                    server_tool_use_data, tool_input=None, is_streaming=True
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

            elif (
                chunk.type == "content_block_start"
                and hasattr(chunk.content_block, "type")
                and chunk.content_block.type == "web_search_tool_result"
            ):
                # Handle web search results in streaming
                self._handle_web_search_tool_result(
                    chunk.content_block, is_streaming=True
                )
                continue

            elif (
                chunk.type == "content_block_start"
                and hasattr(chunk.content_block, "type")
                and chunk.content_block.type == "web_fetch_tool_result"
            ):
                # Handle web fetch results in streaming
                self._handle_web_fetch_tool_result(
                    chunk.content_block, is_streaming=True
                )
                continue

            # Handle tool input JSON parts (for both regular and MCP tools)
            elif (
                chunk.type == "content_block_delta"
                and hasattr(chunk.delta, "type")
                and chunk.delta.type == "input_json_delta"
            ):
                json_input_parts.append(chunk.delta.partial_json)

            # Handle content block stop - parse JSON for server tools here
            elif chunk.type == "content_block_stop":
                # If we have server tool use data and JSON parts, parse and log immediately
                if server_tool_use_data and json_input_parts:
                    try:
                        json_str = "".join(json_input_parts).strip()
                        if json_str:
                            parsed_input = Utility.json_loads(json_str)
                            server_tool_use_data["input"] = parsed_input

                            # Log server tool with parsed input
                            self._handle_server_tool_use(
                                server_tool_use_data,
                                tool_input=parsed_input,
                                is_streaming=True
                            )
                    except json.JSONDecodeError as e:
                        if self.logger.isEnabledFor(logging.ERROR):
                            self.logger.error(f"Error parsing server tool JSON: {e}")

                    # Clear the data and JSON parts for the next tool
                    server_tool_use_data = None
                    json_input_parts = []

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

        # Process JSON input if we have tool use (regular, MCP, or server)
        if (
            tool_use_data or mcp_tool_use_data or server_tool_use_data
        ) and json_input_parts:
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
                    elif server_tool_use_data:
                        server_tool_use_data["input"] = parsed_input
                        # Log server tool with parsed input
                        self._handle_server_tool_use(
                            server_tool_use_data,
                            tool_input=parsed_input,
                            is_streaming=True
                        )
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
                    elif server_tool_use_data:
                        server_tool_use_data["input"] = {}

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
            and kwargs["encoded_content"]
            and file.downloadable
        ):
            response: Response = self.client.beta.files.download(kwargs["file_id"])
            content = response.content  # Get the actual bytes data)
            # Convert the content to a Base64-encoded string
            uploaded_file["encoded_content"] = base64.b64encode(content).decode("utf-8")

        return uploaded_file
