# -*-coding: utf-8 -*-
import os
os.system("ifconfig")


import math
import time
import uuid
import json
import logging

import torch
import argparse
from enum import Enum
from typing import Optional, List, AsyncGenerator
from pydantic import ValidationError

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response, StreamingResponse, JSONResponse
import openai.types.chat.chat_completion as chat_types
from openai.types.completion_usage import CompletionUsage
import openai.types.chat.chat_completion_message_tool_call as chat_message_tool_call_types
import openai.types.chat.chat_completion_chunk as chat_chunk_types
import openai.types.chat.completion_create_params as create_types

from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
from lmformatenforcer import TokenEnforcer, JsonSchemaParser, TokenEnforcerTokenizerData
from transformers import PreTrainedTokenizer, AutoTokenizer
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.utils import random_uuid

# from string_parser_patch import StringParsingState
# lmformatenforcer.jsonschemaparser.StringParsingState = StringParsingState


SYSTEM_TOKEN = "<|SYSTEM|>"
USER_TOKEN = "<|USER|>"
ASSISTANT_TOKEN = "<|ASSISTANT|>"
FUNCTION_CALL_TOKEN = "<|FUNCTION_CALL|>"
PARAMETERS_TOKEN = "<|PARAMETERS|>"
FUNCITON_OUTPUT_TOKEN = "<|FUNCTION_OUTPUT|>"
CONTENT_TOKEN = "<|CONTENT|>"
DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant."
FUNCTION_SUFIX = "你可以使用这些函数来帮助用户解决问题: "


def apply_chat_template(messages, functions, tokenizer, add_generation_token=True):
    text = ""
    if messages[0]["role"] == "system":
        system_msg = f"{SYSTEM_TOKEN}{messages[0]['content']}"
        messages = messages[1:]
    else:
        system_msg = f"{SYSTEM_TOKEN}{DEFAULT_SYSTEM_MESSAGE}"
    if functions:
        system_msg += f"\n\n{FUNCTION_SUFIX}{json.dumps(functions, ensure_ascii=False)}"
    text += system_msg

    tool_rsp_cache = []
    for i, msg in enumerate(messages):
        if msg["role"] == "user":
            text += f"{USER_TOKEN}{msg['content']}"
        elif msg["role"] == "assistant":
            assistant_text = ASSISTANT_TOKEN + (f"{CONTENT_TOKEN}{msg['content']}" if msg["content"] else "")
            fcs = msg.get("tool_calls")
            if fcs:
                for fc in fcs:
                    assistant_text += f"{FUNCTION_CALL_TOKEN}{fc['function']['name']}{PARAMETERS_TOKEN}{json.dumps(fc['function']['arguments'], ensure_ascii=False)}"
            text += assistant_text + tokenizer.eos_token
        elif msg["role"] == "tool":
            # https://platform.openai.com/docs/guides/function-calling#integration-guide:~:text=.-,Parallel%20tool%20calling,-Forcing%20tool%20calls
            # 根据上面链接，多个tool调用结果是多条message，但是目前训练的时候作为一个message了（见hermes.ipynb），先恶心处理
            tool_rsp_cache.append(msg)
            if i == len(messages) - 1 or messages[i + 1]["role"] != "tool":
                contents = []
                for m in tool_rsp_cache:
                    try:
                        contents.append(json.loads(m["content"]))
                    except:
                        contents.append(m["content"])
                text += FUNCITON_OUTPUT_TOKEN + json.dumps(contents, ensure_ascii=False)
    if add_generation_token:
        text += ASSISTANT_TOKEN
    return text


def format_sse(data):
    return f"data: {json.dumps(data.model_dump(), ensure_ascii=False)}\n\n"


app = FastAPI()
engine = None

global tokenizer
global tokenizer_data
global engine_args


class State(Enum):
    DECIDE = 0
    CONTENT = 1
    SELECT = 2
    PARAMS = 3


state_change = {
    State.DECIDE: {
        CONTENT_TOKEN: State.CONTENT,
        FUNCTION_CALL_TOKEN: State.SELECT,
    },
    State.CONTENT: {
        FUNCTION_CALL_TOKEN: State.SELECT,
    },
    State.SELECT: {
        PARAMETERS_TOKEN: State.PARAMS,
    },
    State.PARAMS: {
        FUNCTION_CALL_TOKEN: State.SELECT,
    }
}

class VLLMLogitsProcessor(object):
    def __init__(self, req: create_types.CompletionCreateParams,
                 tokenizer: PreTrainedTokenizer,
                 tokenizer_data: TokenEnforcerTokenizerData,
                 ):
        self._state = None
        self.mask: Optional[torch.Tensor] = None
        self.tokenizer_data = tokenizer_data

        self.req = req
        self.tokenizer = tokenizer

        if not req.get("tools"):
            self.tool_choice = "none"
        else:
            self.tool_choice = "auto"
        if tc := req.get("tool_choice"):
            self.tool_choice = tc

        self.functions = []
        if ts := req.get("tools"):
            self.functions = [t["function"] for t in ts]

        self.select_cache = []
        self.fname_to_id = {}
    
        for f in self.functions:
            name = f["name"]
            self.fname_to_id[name] = tokenizer.encode(name, add_special_tokens=False)

        self.text_cache = ""
        self.id_cache = []
        self.enforcer = None

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, v):
        self._state = v

    def __call__(self, input_ids: List[int], scores: torch.Tensor) -> torch.Tensor:
        if self.mask is not None:
            self.mask.fill_(-math.inf)
        else:
            self.mask = torch.full_like(scores, -math.inf)

        # 状态跳转
        if self.state is None:
            self.state = State.DECIDE
        elif self.state == State.DECIDE:
            if input_ids[-1] == self.tokenizer.convert_tokens_to_ids(FUNCTION_CALL_TOKEN):
                self.state = State.SELECT
            else:
                self.state = State.CONTENT
        elif self.state == State.CONTENT:
            if input_ids[-1] == self.tokenizer.convert_tokens_to_ids(FUNCTION_CALL_TOKEN):
                self.state = State.SELECT
        elif self.state == State.SELECT:
            self.select_cache.append(input_ids[-1])
            for k, v in self.fname_to_id.items():  # 不考虑函数名互为前缀情况
                if self.select_cache == v:
                    self.select_cache = []
                    self.state = State.PARAMS
                    self.text_cache += PARAMETERS_TOKEN
                    for f in self.functions:
                        if f["name"] == k:
                            schema = f["parameters"]
                            try:
                                self.enforcer = TokenEnforcer(self.tokenizer_data, JsonSchemaParser(schema))
                            except Exception as e:
                                logging.exception(e)
                                print(f"error schema: {schema}")
                    break
        elif self.state == State.PARAMS:
            if input_ids[-1] == self.tokenizer.convert_tokens_to_ids(FUNCTION_CALL_TOKEN):
                self.state = State.SELECT

        if not self.id_cache and self.text_cache:
            self.id_cache = self.tokenizer.encode(self.text_cache, add_special_tokens=False)
            self.text_cache = ""
        if self.id_cache:
            direct_id = self.id_cache.pop(0)
            self.mask[direct_id] = 0
            return scores + self.mask

        # 状态具体逻辑
        if self.state == State.DECIDE:
            if isinstance(self.tool_choice, dict) or self.tool_choice == "required":
                self.mask[self.tokenizer.convert_tokens_to_ids(FUNCTION_CALL_TOKEN)] = 0
            elif self.tool_choice == "none":
                self.mask[self.tokenizer.convert_tokens_to_ids(CONTENT_TOKEN)] = 0
            else:  # auto
                self.mask[self.tokenizer.convert_tokens_to_ids(FUNCTION_CALL_TOKEN)] = 0
                self.mask[self.tokenizer.convert_tokens_to_ids(CONTENT_TOKEN)] = 0
        elif self.state == State.CONTENT:
            self.mask.fill_(0.)
            if self.tool_choice == "none":
                self.mask[self.tokenizer.convert_tokens_to_ids(FUNCTION_CALL_TOKEN)] = -math.inf
            self.mask[self.tokenizer.convert_tokens_to_ids([
                SYSTEM_TOKEN,
                USER_TOKEN,
                ASSISTANT_TOKEN,
                PARAMETERS_TOKEN,
                FUNCITON_OUTPUT_TOKEN,
                CONTENT_TOKEN,
            ])] = -math.inf
        elif self.state == State.SELECT:
            allowed_ids = set()
            if not self.select_cache:
                if isinstance(self.tool_choice, dict):
                    fname = self.tool_choice["function"]["name"]
                    self.id_cache.extend(self.fname_to_id[fname][1:])
                    allowed_ids.add(self.fname_to_id[fname][0])
                else:
                    for token_ids in self.fname_to_id.values():
                        allowed_ids.add(token_ids[0])
            else:
                for token_ids in self.fname_to_id.values():
                    if len(token_ids) > len(self.select_cache):
                        if token_ids[:len(self.select_cache)] == self.select_cache:
                            allowed_ids.add(token_ids[len(self.select_cache)])
            self.mask[list(allowed_ids)] = 0
        else:  # self.state == State.PARAMS
            try:
                allowed_tokens = self.enforcer.get_allowed_tokens(input_ids)
                if self.tokenizer.eos_token_id in allowed_tokens:
                    allowed_tokens.append(self.tokenizer.convert_tokens_to_ids(FUNCTION_CALL_TOKEN))  # support paralle fc call
            except Exception as e:
                print(e)
                logging.exception(e)
            self.mask[allowed_tokens] = 0
        return scores + self.mask


async def stream_results(results_generator, stream: bool) -> AsyncGenerator[
    bytes, None]:
    gen_text = ""
    gen_token_ids = []
    rid = str(uuid.uuid4())
    model = "vllm"
    result = {"content": "", "tool_calls": []}
    tool_call_cache = {"name": "", "arguments": ""}
    state = State.DECIDE

    async for request_output in results_generator:
        # print(f"gen_text: {request_output.outputs[0].text}")
        # print(f"gen_ids: {request_output.outputs[0].token_ids}")
        chunk = request_output.outputs[0].text[len(gen_text):]
        token_id = request_output.outputs[0].token_ids[len(gen_token_ids):][0]
        # print(state, chunk if chunk else tokenizer.convert_ids_to_tokens(token_id)[0])
        gen_text = request_output.outputs[0].text
        gen_token_ids = request_output.outputs[0].token_ids


        if tokenizer.convert_ids_to_tokens(token_id) in (FUNCTION_CALL_TOKEN, tokenizer.eos_token) and tool_call_cache["name"]:
            result["tool_calls"].append(tool_call_cache)
            tool_call_cache = {"name": "", "arguments": ""}

        if next_state := state_change[state].get(tokenizer.convert_ids_to_tokens(token_id)):
            state = next_state
            # print(f"-> {state}")
            continue

        if state == State.CONTENT:
            result["content"] += chunk
            yield format_sse(chat_chunk_types.ChatCompletionChunk(
                id=rid,
                choices=[chat_chunk_types.Choice(
                    index=0,
                    delta=chat_chunk_types.ChoiceDelta(content=chunk),
                )],
                created=int(time.time()),
                model=model,
                object="chat.completion.chunk"
            ))
        elif state == State.SELECT:
            tool_call_cache["name"] += chunk
        elif state == State.PARAMS and token_id != tokenizer.eos_token_id:
            if not tool_call_cache["arguments"]:  # only once
                yield format_sse(chat_chunk_types.ChatCompletionChunk(
                    id=rid,
                    choices=[chat_chunk_types.Choice(
                        index=0,
                        delta=chat_chunk_types.ChoiceDelta(
                            tool_calls=[chat_chunk_types.ChoiceDeltaToolCall(
                                index=len(result["tool_calls"]),
                                function=chat_chunk_types.ChoiceDeltaToolCallFunction(name=tool_call_cache["name"])
                            )]
                        )
                    )],
                    created=int(time.time()),
                    model=model,
                    object="chat.completion.chunk"
                ))

            tool_call_cache["arguments"] += chunk
            yield format_sse(chat_chunk_types.ChatCompletionChunk(
                id=rid,
                choices=[chat_chunk_types.Choice(
                    index=0,
                    delta=chat_chunk_types.ChoiceDelta(
                        tool_calls=[chat_chunk_types.ChoiceDeltaToolCall(
                            index=len(result["tool_calls"]),
                            function=chat_chunk_types.ChoiceDeltaToolCallFunction(arguments=chunk)
                        )]
                    )
                )],
                created=int(time.time()),
                model=model,
                object="chat.completion.chunk"
            ))

    if state == State.PARAMS:
        if stream:
            yield format_sse(chat_chunk_types.ChatCompletionChunk(
                id=rid,
                choices=[chat_chunk_types.Choice(
                    index=0,
                    delta=chat_chunk_types.ChoiceDelta(),
                    finish_reason="tool_calls"
                )],
                created=int(time.time()),
                model=model,
                object="chat.completion.chunk"
            ))
        else:
            yield chat_types.ChatCompletion(
                id=rid,
                choices=[chat_types.Choice(
                    finish_reason="tool_calls",
                    index=0,
                    message=chat_types.ChatCompletionMessage(
                        content=result["content"] if result["content"] else None,
                        tool_calls=[chat_message_tool_call_types.ChatCompletionMessageToolCall(
                            id=random_uuid(),
                            type="function",
                            function=chat_message_tool_call_types.Function(
                                name=r["name"], arguments=r["arguments"]
                            )
                        ) for r in result["tool_calls"]],
                        role="assistant"
                    )
                )],
                created=int(time.time()),
                usage=CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),  # 为了bfcl不报错
                model=model,
                object="chat.completion"
            )
    else:
        if stream:
            yield format_sse(chat_chunk_types.ChatCompletionChunk(
                id=rid,
                choices=[chat_chunk_types.Choice(
                    index=0,
                    delta=chat_chunk_types.ChoiceDelta(),
                    finish_reason="stop"
                )],
                created=int(time.time()),
                model=model,
                object="chat.completion.chunk"
            ))
        else:
            yield chat_types.ChatCompletion(
                id=rid,
                choices=[chat_types.Choice(
                    finish_reason="stop",
                    index=0,
                    message=chat_types.ChatCompletionMessage(
                        content=result["content"],
                        role="assistant"
                    )
                )],
                created=int(time.time()),
                usage=CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                model=model,
                object="chat.completion"
            )


@app.post("/chat/completions")
async def chat_completion(request: Request) -> Response:
    data = await request.json()
    data.setdefault("stream", False)

    try:
        req: create_types.CompletionCreateParams = data
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    functions = []
    if tools := req.get("tools"):
        functions = [t["function"] for t in tools]
        
    prompt = apply_chat_template(req["messages"], functions, tokenizer)
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    print(f"req tokens: {len(token_ids)}")
    
    sampling_params = SamplingParams(
        max_tokens=req.get("max_tokens", 1024), 
        temperature=req.get("temperature", 0.3),
        presence_penalty=req.get("presence_penalty", 0.15),
        frequency_penalty=req.get("frequency_penalty", 0.15), 
        repetition_penalty=1.05,
    )
    processor = VLLMLogitsProcessor(req, tokenizer, tokenizer_data)
    sampling_params.logits_processors = [processor]
    request_id = random_uuid()

    try:
        results_generator = engine.generate(
            prompt={"prompt_token_ids": token_ids}, 
            sampling_params=sampling_params, 
            request_id=request_id
        )
        generator = stream_results(results_generator, req["stream"])
        if req["stream"]:
            return StreamingResponse(generator, media_type="text/event-stream")
        else:
            final_output = None
            async for request_output in generator:
                if await request.is_disconnected():
                    await engine.abort(request_id)
                    return Response(status_code=499)
                final_output = request_output

            assert final_output is not None
            return JSONResponse(final_output.dict())
    except Exception as e:
        logging.exception(e)
        return JSONResponse({"error": "unknown error"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    tokenizer = AutoTokenizer.from_pretrained(
        engine_args.model,
        use_fast=False,
    )
    tokenizer_data = build_token_enforcer_tokenizer_data(tokenizer)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug",
        timeout_keep_alive=5
    )
