# Qwen3 Tool Call Server

A Qwen2.5 and Qwen3 LLM server implemented with FastAPI and vLLM, providing fully OpenAI API-compatible tool call functionality with complete support for tool choice and structured output.

## Features

- 🚀 High-performance implementation based on FastAPI and vLLM
- 🔧 Complete Tool Calls support
- 🌊 Support for both streaming and non-streaming output
- 🧠 Support for Qwen3 model's reasoning capabilities
- ⚡ Asynchronous processing for high-performance responses
- 🎯 Fully compatible with OpenAI API specifications

## Tool Calls Functionality

### Core Features

- **Complete Tool Choice Support**
  - `auto`: Automatic tool selection
  - `none`: Disable tool calls
  - `required`: Force tool usage
  - Specify specific tool names

- **Parallel Tool Calls**
  - Support for multiple simultaneous tool calls
  - Maintains strict output format consistency

- **Structured Output**
  - Full support for OpenAI's strict mode
  - Maintains strict output format even in parallel tool call scenarios
  - Surpasses limitations of OpenAI's official implementation (e.g., [strict mode cannot be enabled in parallel function calling scenarios](https://platform.openai.com/docs/guides/function-calling/parallel-function-calling?api-mode=responses#:~:text=Note%3A%20Currently%2C%20if%20the%20model%20calls%20multiple%20functions%20in%20one%20turn%20then%20strict%20mode%20will%20be%20disabled%20for%20those%20calls.))

- **Qwen3 Special Features**
  - Enable reasoning functionality through the `reasoning_effort` parameter (any value works)
  - Maintains interface consistency with OpenAI API

## Installation

```bash
# Install dependencies
uv sync

# Start the server
bash run.sh
```

## Usage

### API Examples

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000",
    api_key="not-needed"
)

# Basic chat
response = client.chat.completions.create(
    model="qwen",
    messages=[
        {"role": "user", "content": "Hello"}
    ]
)

# Using tool calls
response = client.chat.completions.create(
    model="qwen",
    messages=[
        {"role": "user", "content": "Check the weather"}
    ],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information for a specific city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["city"]
            }
        }
    }]
)

# Enable Qwen3's reasoning functionality, compatible with tool calls
response = client.chat.completions.create(
    model="qwen",
    messages=[
        {"role": "user", "content": "Analyze this problem"}
    ],
    reasoning_effort="low"  # Any value works
)
```

## Technical Architecture

- **Web Framework**: FastAPI
- **Inference Engine**: vLLM
- **Grammar Control**: xgrammar
- **Async Processing**: asyncio
- **Model Support**: Qwen2.5/Qwen3 series models 