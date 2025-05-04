# Qwen3 Tool Call Server

基于 FastAPI 和 vLLM 实现的 Qwen2.5 和 Qwen3 LLM 服务器，提供与 OpenAI API 完全兼容的 tool call 功能，支持完整的 tool choice 和 structured output。

## 特性

- 🚀 基于 FastAPI 和 vLLM 的高性能实现
- 🔧 完整的工具调用（Tool Calls）支持
- 🌊 支持流式输出和非流式输出
- 🧠 支持 Qwen3 模型的 reasoning 功能
- ⚡ 异步处理，高性能响应
- 🎯 完全兼容 OpenAI API 规范

## 工具调用（Tool Calls）功能

### 核心特性

- **完整的 Tool Choice 支持**
  - `auto`: 自动选择工具
  - `none`: 禁用工具调用
  - `required`: 强制使用工具
  - 指定具体工具名称

- **并行工具调用**
  - 支持多个工具同时调用
  - 保持输出格式的严格一致性

- **Structured Output**
  - 完全支持 OpenAI 的 strict 模式
  - 在并行工具调用场景下仍保持严格输出格式
  - 超越 OpenAI 官方实现的限制（比如[并行工具调用场景下不能开启 strict 模式](https://platform.openai.com/docs/guides/function-calling/parallel-function-calling?api-mode=responses#:~:text=Note%3A%20Currently%2C%20if%20the%20model%20calls%20multiple%20functions%20in%20one%20turn%20then%20strict%20mode%20will%20be%20disabled%20for%20those%20calls.)）

- **Qwen3 特殊功能**
  - 为保持与 OpenAI API 的接口一致性，通过 `reasoning_effort` 参数启用 reasoning 功能（传任意值均可）

## 安装

```bash
# 安装依赖
uv sync

# 启动服务器
bash run.sh
```

## 使用方法

### API 使用示例

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000",
    api_key="not-needed"
)

# 普通聊天
response = client.chat.completions.create(
    model="qwen",
    messages=[
        {"role": "user", "content": "你好"}
    ]
)

# 使用工具调用
response = client.chat.completions.create(
    model="qwen",
    messages=[
        {"role": "user", "content": "查询天气"}
    ],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称"
                    }
                },
                "required": ["city"]
            }
        }
    }]
)

# 启用 Qwen3 的 reasoning 功能，与 tool call 兼容
response = client.chat.completions.create(
    model="qwen",
    messages=[
        {"role": "user", "content": "分析这个问题"}
    ],
    reasoning_effort="low"  # 任意值均可
)
```

## 技术架构

- **Web 框架**: FastAPI
- **推理引擎**: vLLM
- **语法控制**: xgrammar
- **异步处理**: asyncio
- **模型支持**: Qwen2.5/Qwen3 系列模型

## BFCL 测试结果

| Overall Acc | Model | Non-Live AST Acc | Non-Live Simple AST | Non-Live Multiple AST | Non-Live Parallel AST | Non-Live Parallel Multiple AST | Live Acc | Live Simple AST | Live Multiple AST | Live Parallel AST | Live Parallel Multiple AST | Multi Turn Acc | Multi Turn Base | Multi Turn Miss Func | Multi Turn Miss Param | Multi Turn Long Context | Relevance Detection | Irrelevance Detection |
|------------|-------|------------------|---------------------|----------------------|----------------------|-------------------------------|----------|-----------------|-------------------|-------------------|--------------------------|----------------|-----------------|---------------------|----------------------|------------------------|-------------------|----------------------|
| 65.44% | Qwen3-32b-reasoning | 85.67% | 74.67% | 95.50% | 84.50% | 88.00% | 77.17% | 82.56% | 75.97% | 87.50% | 66.67% | 34.12% | 38.50% | 41.00% | 30.50% | 26.50% | 66.67% | 79.91% |
| 61.64% | Qwen3-32b | 87.44% | 74.75% | 91.50% | 92.00% | 91.50% | 78.10% | 79.84% | 78.92% | 68.75% | 75.00% | 20.62% | 31.00% | 13.50% | 20.00% | 18.00% | 83.33% | 79.00% |
| 61.19% | Qwen2.5-32b | 85.54% | 71.67% | 92.00% | 92.00% | 86.50% | 77.43% | 76.36% | 75.69% | 50.00% | 58.33% | 21.12% | 28.00% | 22.50% | 21.50% | 12.50% | 72.22% | 81.93% |
| 58.47% | Qwen2.5-7b | 88.50% | 77.00% | 96.00% | 91.50% | 89.50% | 74.86% | 75.58% | 72.74% | 62.50% | 79.17% | 12.50% | 17.50% | 13.00% | 11.50% | 8.00% | 88.89% | 81.62% |
