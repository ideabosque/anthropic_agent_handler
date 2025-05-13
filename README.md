# 🧠 AnthropicEventHandler

The `AnthropicEventHandler` is a concrete implementation of the `AIAgentEventHandler` base class designed to interface with Anthropic Claude models. It orchestrates message formatting, model invocation, tool integration, streaming, and threading within the AI agent execution pipeline.

This handler enables a **stateless, multi-turn AI orchestration** system built to support tools like `get_weather_forecast`.

---

## 🖉 Inheritance

![AI Agent Event Handler Class Diagram](/images/ai_agent_event_handler_class_diagram.jpg)

```
AIAgentEventHandler
     ▲
     └── AnthropicEventHandler
```

---

## 📦 Module Features

### 🔧 Attributes

* `client`: Anthropic Claude API client instance
* `model_settings`: A dictionary containing Claude model configuration (e.g., `model`, `temperature`, `tools`, etc.)

### 📞 Core Method: `invoke_model`

```python
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
```

---

## 📘 Sample Configuration (Anthropic)

```json
{
  "instructions": "# 📚 System Prompt Instructions: Weather Assistant

You are an AI Assistant responsible for answering user queries about the weather using the `get_weather_forecast` function. Analyze user input to extract the city and date. Retrieve the relevant weather data and respond clearly and concisely.",
  "functions": {
    "get_weather_forecast": {
      "class_name": "WeatherForecastFunction",
      "module_name": "weather_funct",
      "configuration": {}
    }
  },
  "function_configuration": {
    "endpoint_id": "anthropic",
    "region_name": "${region_name}",
    "aws_access_key_id": "${aws_access_key_id}",
    "aws_secret_access_key": "${aws_secret_access_key}"
  },
  "configuration": {
    "model": "claude-3-opus-20240229",
    "api_key": "${api_key}",
    "temperature": 0,
    "tools": [
      {
        "type": "function",
        "name": "get_weather_forecast",
        "description": "Get weather forecast by city and date",
        "strict": true,
        "parameters": {
          "type": "object",
          "properties": {
            "city": {
              "type": "string",
              "description": "The city name."
            },
            "date": {
              "type": "string",
              "description": "The date to check weather for (YYYY-MM-DD)."
            }
          },
          "required": ["city", "date"],
          "additionalProperties": false
        }
      }
    ]
  },
  "num_of_messages": 30,
  "tool_call_role": "assistant"
}
```
---

---
## 💬 Full-Scale Chatbot Scripts

### 🔁 Non-Streaming Chatbot Script

```python
import logging
import os
import sys

import pendulum
from dotenv import load_dotenv
from ai_agent_handler import AIAgentEventHandler
from anthropic_agent_handler import AnthropicEventHandler

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()

load_dotenv()
setting = {
    "region_name": os.getenv("region_name"),
    "aws_access_key_id": os.getenv("aws_access_key_id"),
    "aws_secret_access_key": os.getenv("aws_secret_access_key"),
    "funct_bucket_name": os.getenv("funct_bucket_name"),
    "funct_zip_path": os.getenv("funct_zip_path"),
    "funct_extract_path": os.getenv("funct_extract_path"),
    "connection_id": os.getenv("connection_id"),
    "endpoint_id": os.getenv("endpoint_id"),
    "test_mode": os.getenv("test_mode"),
}

weather_agent = { ... }  # Configuration as defined above
handler = AnthropicEventHandler(logger=None, agent=weather_agent, **setting)
handler.short_term_memory = []

def get_input_messages(messages, num_of_messages):
    return [msg["message"] for msg in sorted(messages, key=lambda x: x["created_at"], reverse=True)][:num_of_messages][::-1]

while True:
    user_input = input("User: ")
    if user_input.strip().lower() in ["exit", "quit"]:
        print("Chatbot: Goodbye!")
        break

    message = {"role": "user", "content": user_input}
    handler.short_term_memory.append({"message": message, "created_at": pendulum.now("UTC")})
    messages = get_input_messages(handler.short_term_memory, weather_agent["num_of_messages"])
    run_id = handler.ask_model(messages)

    print("Chatbot:", handler.final_output["content"])
    handler.short_term_memory.append({
        "message": handler.final_output,
        "created_at": pendulum.now("UTC")
    })
```

### 🔁 Streaming Chatbot Script

```python
import logging
import os
import sys

import pendulum
from dotenv import load_dotenv
from ai_agent_handler import AIAgentEventHandler
from anthropic_agent_handler import AnthropicEventHandler

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()

load_dotenv()
setting = {
    "region_name": os.getenv("region_name"),
    "aws_access_key_id": os.getenv("aws_access_key_id"),
    "aws_secret_access_key": os.getenv("aws_secret_access_key"),
    "funct_bucket_name": os.getenv("funct_bucket_name"),
    "funct_zip_path": os.getenv("funct_zip_path"),
    "funct_extract_path": os.getenv("funct_extract_path"),
    "connection_id": os.getenv("connection_id"),
    "endpoint_id": os.getenv("endpoint_id"),
    "test_mode": os.getenv("test_mode"),
}

weather_agent = { ... }  # Configuration as defined above
handler = AnthropicEventHandler(logger=None, agent=weather_agent, **setting)
handler.short_term_memory = []

def get_input_messages(messages, num_of_messages):
    return [msg["message"] for msg in sorted(messages, key=lambda x: x["created_at"], reverse=True)][:num_of_messages][::-1]

while True:
    user_input = input("User: ")
    if user_input.strip().lower() in ["exit", "quit"]:
        print("Chatbot: Goodbye!")
        break

    message = {"role": "user", "content": user_input}
    handler.short_term_memory.append({"message": message, "created_at": pendulum.now("UTC")})
    messages = get_input_messages(handler.short_term_memory, weather_agent["num_of_messages"])

    stream_queue = Queue()
    stream_event = threading.Event()
    stream_thread = threading.Thread(
        target=handler.ask_model,
        args=[messages, stream_queue, stream_event],
        daemon=True
    )
    stream_thread.start()

    result = stream_queue.get()
    if result["name"] == "run_id":
        print("Run ID:", result["value"])

    stream_event.wait()
    print("Chatbot:", handler.final_output["content"])
    handler.short_term_memory.append({
        "message": handler.final_output,
        "created_at": pendulum.now("UTC")
    })
```

---