#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

__author__ = "bibow"

import json
import logging
import threading
import traceback
import uuid
from decimal import Decimal
from queue import Queue
from typing import Any, Dict, List, Optional

import anthropic
import pendulum

from ai_agent_handler import AIAgentEventHandler
from silvaengine_utility import Utility


# ----------------------------
# Anthropic Response Streaming with Function Handling and History
# ----------------------------
class AnthropicEventHandler(AIAgentEventHandler):
    """
    Handles conversations and function calls with the Anthropic API by:
    - Managing streaming text responses
    - Processing function calls in responses
    - Executing functions and handling results
    - Maintaining conversation history
    - Storing final outputs
    """

    def __init__(
        self,
        logger: logging.Logger,
        agent: Dict[str, Any],
        **setting: Dict[str, Any],
    ) -> None:
        """
        Initialize the Anthropic event handler

        Args:
            logger: Logger for debug/info messages
            agent: Agent configuration and tools
            setting: Additional handler settings
        """
        AIAgentEventHandler.__init__(self, logger, agent, **setting)

        self.logger = logger
        self.client = anthropic.Anthropic(api_key=agent["configuration"].get("api_key"))
        self.model_setting = dict(
            {
                k: float(v) if isinstance(v, Decimal) else v
                for k, v in agent["configuration"].items()
                if k not in ["api_key", "text"]
            },
            **{
                "system": agent["instructions"],
            },
        )
        self.assistant_messages = []

    def invoke_model(self, **kwargs: Dict[str, Any]) -> Any:
        """
        Call the Anthropic API with messages and handle responses

        Args:
            kwargs: Input messages and streaming configuration

        Returns:
            Model response object

        Raises:
            Exception: If API call fails
        """
        try:
            messages = list(
                filter(lambda x: bool(x["content"]) == True, kwargs["input"])
            )
            return self.client.messages.create(
                **dict(
                    self.model_setting,
                    **{"messages": messages, "stream": kwargs["stream"]},
                )
            )
        except Exception as e:
            self.logger.error(f"Error invoking model: {str(e)}")
            raise Exception(f"Failed to invoke model: {str(e)}")

    def ask_model(
        self,
        input_messages: List[Dict[str, Any]],
        queue: Queue = None,
        stream_event: threading.Event = None,
        model_setting: Dict[str, Any] = None,
    ) -> Optional[str]:
        """
        Send request to Anthropic API with streaming or non-streaming mode

        Args:
            input_messages: Conversation history and current question
            queue: Queue for streaming events, enables streaming if provided
            stream_event: Event to signal streaming completion
            model_setting: Optional model configuration overrides

        Returns:
            Response ID for non-streaming requests, None for streaming
        """
        try:
            if not self.client:
                self.logger.error("No Anthropic client provided.")
                return None

            stream = True if queue is not None else False

            # Add model-specific settings if provided
            if model_setting:
                self.model_setting.update(model_setting)

            timestamp = pendulum.now("UTC").int_timestamp
            run_id = f"run-antropic-{self.model_setting['model']}-{timestamp}-{str(uuid.uuid4())[:8]}"

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

            self.handle_output(response, input_messages)
            return run_id

        except Exception as e:
            self.logger.error(f"Error in ask_model: {str(e)}")
            raise Exception(f"Failed to process model request: {str(e)}")

    def handle_function_call(
        self,
        tool_call: Any,
        input_messages: List[Dict[str, Any]],
        stream_event: threading.Event = None,
    ) -> None:
        """
        Process and execute function calls from model responses by:
        1. Extracting function details
        2. Running the function with arguments
        3. Recording execution results
        4. Updating conversation history
        5. Continuing conversation flow

        Args:
            tool_call: Function call details from model
            input_messages: Conversation history to update
            stream_event: Event for streaming completion

        Raises:
            ValueError: For invalid tool calls
            Exception: For function execution failures
        """
        try:
            # Extract function call metadata
            function_call_data = {
                "id": tool_call["id"],
                "arguments": tool_call["input"],
                "name": tool_call["name"],
                "type": tool_call["type"],
            }

            # Record initial function call
            self.logger.info(
                f"[handle_function_call] Starting function call recording for {function_call_data['name']}"
            )
            self._record_function_call_start(function_call_data)

            # Parse and process arguments
            self.logger.info(
                f"[handle_function_call] Processing arguments for function {function_call_data['name']}"
            )
            arguments = self._process_function_arguments(function_call_data)

            # Execute function and handle result
            self.logger.info(
                f"[handle_function_call] Executing function {function_call_data['name']} with arguments {arguments}"
            )
            function_output = self._execute_function(function_call_data, arguments)

            # Update conversation history
            self.logger.info(
                f"[handle_function_call][{function_call_data['name']}] Updating conversation history"
            )
            self._update_conversation_history(
                function_call_data, function_output, input_messages
            )

            # Continue conversation
            self.logger.info(
                f"[handle_function_call][{function_call_data['name']}] Continuing conversation"
            )
            self._continue_conversation(input_messages, stream_event)

            if self._run is None:
                self._short_term_memory.append(
                    {
                        "message": {
                            "role": self.agent["tool_call_role"],
                            "content": Utility.json_dumps(
                                {
                                    "tool": {
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

        except Exception as e:
            self.logger.error(f"Error in handle_function_call: {e}")
            raise

    def _record_function_call_start(self, function_call_data: Dict[str, Any]) -> None:
        """
        Store initial function call metadata in async storage

        Args:
            function_call_data: Function call details to record
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
        Process and validate function arguments

        Args:
            function_call_data: Raw function call data

        Returns:
            Processed arguments with endpoint ID

        Raises:
            ValueError: If argument processing fails
        """
        try:
            arguments = function_call_data.get("arguments", {})
            arguments["endpoint_id"] = self._endpoint_id

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
            self.logger.error("Error parsing function arguments: %s", e)
            raise ValueError(f"Failed to parse function arguments: {e}")

    def _execute_function(
        self, function_call_data: Dict[str, Any], arguments: Dict[str, Any]
    ) -> Any:
        """
        Execute requested function and handle results

        Args:
            function_call_data: Function metadata
            arguments: Processed function arguments

        Returns:
            Function output or error message

        Raises:
            ValueError: If function not supported
        """
        agent_function = self.get_function(function_call_data["name"])
        if not agent_function:
            raise ValueError(
                f"Unsupported function requested: {function_call_data['name']}"
            )

        try:
            self.invoke_async_funct(
                "async_insert_update_tool_call",
                **{
                    "tool_call_id": function_call_data["id"],
                    "arguments": Utility.json_dumps(arguments),
                    "status": "in_progress",
                },
            )

            function_output = agent_function(**arguments)

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
            self.invoke_async_funct(
                "async_insert_update_tool_call",
                **{
                    "tool_call_id": function_call_data["id"],
                    "arguments": Utility.json_dumps(arguments),
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
        Add function call results to conversation history

        Args:
            function_call_data: Function call details
            function_output: Function execution results
            input_messages: Conversation history to update
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

    def _continue_conversation(
        self,
        input_messages: List[Dict[str, Any]],
        stream_event: threading.Event = None,
    ) -> None:
        """
        Resume conversation after function execution

        Args:
            input_messages: Updated conversation history
            stream_event: Event for streaming completion
        """
        response = self.invoke_model(
            **{
                "input": input_messages,
                "stream": bool(stream_event),
            }
        )

        if stream_event:
            self.handle_stream(response, input_messages, stream_event)
        else:
            self.handle_output(response, input_messages)

    def handle_output(
        self,
        response: Any,
        input_messages: List[Dict[str, Any]],
    ) -> None:
        """
        Process single response from model

        Args:
            response: Model response object
            input_messages: Conversation history to update
        """
        self.logger.info("Processing output: %s", response)

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
                    self.handle_function_call(tool_call, input_messages)
        else:
            content = response.content[0].text
            while self.assistant_messages:
                assistant_message = self.assistant_messages.pop()
                content = assistant_message["content"] + " " + content

            self.final_output = {
                "message_id": response.id,
                "role": response.role,
                "content": content,
            }

    def handle_stream(
        self,
        response_stream,
        input_messages: List[Dict[str, Any]] = None,
        stream_event: threading.Event = None,
    ) -> None:
        """
        Process streaming response chunks from model

        Args:
            response_stream: Streaming response iterator
            input_messages: Conversation history to update
            stream_event: Event to signal streaming completion
        """
        message_id = None
        json_input_parts = []
        stop_reason = None
        tool_use_data = None
        self.accumulated_text = ""
        accumulated_partial_json = ""
        accumulated_partial_text = ""
        output_format = (
            self.model_setting.get("text", {"format": {"type": "text"}})
            .get("format", {"type": "text"})
            .get("type", "text")
        )
        index = 0
        if self.assistant_messages:
            index = self.assistant_messages[-1]["index"]
            self.send_data_to_websocket(
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
                else:
                    self.accumulated_text += chunk.delta.text
                    accumulated_partial_text += chunk.delta.text
                    # Send incremental text chunk to WebSocket server
                    if len(accumulated_partial_text) >= int(
                        self.setting.get("accumulated_partial_text_buffer", "10")
                    ):
                        self.send_data_to_websocket(
                            index=index,
                            data_format=output_format,
                            chunk_delta=accumulated_partial_text,
                        )
                        accumulated_partial_text = ""
                        index += 1

            # Handle tool use start
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

            # Handle tool input JSON parts
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
                    self.send_data_to_websocket(
                        index=index,
                        data_format=output_format,
                        chunk_delta=accumulated_partial_text,
                    )
                    accumulated_partial_text = ""
                    index += 1

                self.send_data_to_websocket(
                    index=index,
                    data_format=output_format,
                    is_message_end=True,
                )

        # Process JSON input if we have tool use
        if tool_use_data and json_input_parts:
            try:
                # Join JSON parts and parse
                json_str = "".join(json_input_parts)
                tool_use_data["input"] = Utility.json_loads(json_str)
            except json.JSONDecodeError:
                print("\nError parsing tool input JSON")

        # Handle tool usage if detected
        if stop_reason == "tool_use" and tool_use_data:
            input_messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": self.accumulated_text},
                        tool_use_data,
                    ],
                }
            )
            self.assistant_messages.append(
                {
                    "content": self.accumulated_text,
                    "index": index,
                }
            )
            self.handle_function_call(
                tool_use_data, input_messages, stream_event=stream_event
            )
            return

        while self.assistant_messages:
            assistant_message = self.assistant_messages.pop()
            self.accumulated_text = (
                assistant_message["content"] + " " + self.accumulated_text
            )

        self.final_output = {
            "message_id": message_id,
            "role": "assistant",
            "content": self.accumulated_text,
        }

        # Signal that streaming has finished
        if stream_event:
            stream_event.set()
