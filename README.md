# octupus-tool-call

[[English](https://github.com/kongjiellx/octupus-tool-call/blob/main/README-en.md)|中文]

## 项目简介
这是一个专注于训练tool call（function calling）模型的项目，将把模型训练过程的所有数据、代码、模型、及推理方案都开源。

## 项目目标&测评方案
目前以[BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html)作为测评基准。本项目以尽量提高模型在BFCL上的评分为目标，会持续迭代、并将每个版本相关改进同步到这里。

## function calling
介绍一下function calling，这是一种让LLM可以输出函数调用指令的方法。LLM作为语言模型，训练过程中接受各种格式文本训练，趋向于使用自然语言回复用户，一般而言输出格式化函数调用指令能力较弱。

本项目主要从两个方面对这个问题进行优化，一是通过function calling特定格式数据对LLM进行训练，二是在推理时对格式进行约束。

## 整体方案要点
1. 基于qwen2.5系列的pretrain模型，使用开源对话数据和function calling数据进行指令微调。不使用instruct模型是因为需要改变chat template，为了方便推理时进行约束，我们使用了一些特殊token。
2. 基于[lm-format-enforcer](https://github.com/noamgat/lm-format-enforcer)实现输出格式化，保证模型生成的函数调用指令是合法的。
3. 兼容openai api的server，支持tool_choice、parallel tool calls、并保证输出的合法性

## 目录介绍
- utils/edit_tokenizer_and_model.py 编辑词表，增加特殊token，并且使用相关的token进行初始化
- run 模型训练脚本
    - train_stg1.sh
        - 仅使用chat格式数据训练
    - train_stg2.sh
        - 在stg1基础上继续训练，混合chat和function call数据训练
- inference
    - openai兼容的模型服务，使用structured outputs技术
    - 使用方法：python oai_server.py --model /your/model --tensor-parallel-size 1 --max-model-len 8192

## 测评方法
1. 使用oai_server.py启动模型服务
2. 无需更改，直接使用bfcl进行测试，服务本身不需要传模型名，但为了走bfcl代码中openai fc模型的逻辑，填写支持的gpt fc模型名即可
```
OPENAI_BASE_URL=http://localhost:8000 bfcl generate --model gpt-4-turbo-2024-04-09-FC --num-threads 8 --test-category all
```


## 迭代记录
### V1

#### 数据
- stg1
    - 使用qwen2.5-72b-instruct重新回答了[open-hermes](https://huggingface.co/datasets/teknium/OpenHermes-2.5)
    - [wild-chat](https://huggingface.co/datasets/allenai/WildChat)里的中英数据
- stg2
    - [hermes-function-calling](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1)
    - [自己构造的中文长function calling数据](https://huggingface.co/datasets/hqfx/fc_zh_hard)
    - function call数据都经过了一些筛选逻辑，具体代码见data目录

#### 模型和数据地址
https://huggingface.co/collections/hqfx/hqfx-octupus-tool-call-v1-6752bc1b3d5dc4e06f394e59

#### 测评结果
| Overall Acc | Model                        | Non-Live AST Acc | Non-Live Simple AST | Non-Live Multiple AST | Non-Live Parallel AST | Non-Live Parallel Multiple AST | Non-Live Exec Acc | Non-Live Simple Exec | Non-Live Multiple Exec | Non-Live Parallel Exec | Non-Live Parallel Multiple Exec | Live Acc | Live Simple AST | Live Multiple AST | Live Parallel AST | Live Parallel Multiple AST | Multi Turn Acc | Multi Turn Base | Multi Turn Miss Func | Multi Turn Miss Param | Multi Turn Long Context | Relevance Detection | Irrelevance Detection |
|-------------|-----------------------------|------------------|---------------------|-----------------------|-----------------------|-------------------------------|------------------|---------------------|-----------------------|-----------------------|-------------------------------|---------|----------------|------------------|------------------|--------------------------|---------------|----------------|--------------------|--------------------|-----------------------|------------------|------------------|
| 52.21%      | hqfx/octupus-tool-call-v1 | 80.69%           | 65.25%              | 94.00%               | 82.50%               | 81.00%                         | 83.70%           | 88.29%              | 92.00%               | 82.00%               | 72.50%                         | 73.92%  | 66.28%         | 69.72%           | 31.25%           | 33.33%                  | 0.12%         | 0.00%          | 0.00%              | 0.50%              | 0.00%                | 73.17%           | 84.46%           |

在[BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html)上面可以排到30名，微幅超过gpt-3.5，multi_turn基本得了0分，还没仔细分析，后续再重点优化。

### V2

#### 数据
- stg2增加了清洗过的[ToolAce](https://huggingface.co/datasets/Team-ACE/ToolACE)数据，清洗脚本见data目录
- 把数据格式都改成了openai的tool calls格式，训练和推理也随之进行适配

#### 推理服务
- 简化了代码，解决了一些之前的bug，如支持函数名互为前缀的情况

#### 模型和数据地址
https://huggingface.co/collections/hqfx/octupus-tool-call-v2-676fa1c11c48eff17fa1c017

#### 测评结果
| Overall Acc | Model                        | Non-Live AST Acc | Non-Live Simple AST | Non-Live Multiple AST | Non-Live Parallel AST | Non-Live Parallel Multiple AST | Non-Live Exec Acc | Non-Live Simple Exec | Non-Live Multiple Exec | Non-Live Parallel Exec | Non-Live Parallel Multiple Exec | Live Acc | Live Simple AST | Live Multiple AST | Live Parallel AST | Live Parallel Multiple AST | Multi Turn Acc | Multi Turn Base | Multi Turn Miss Func | Multi Turn Miss Param | Multi Turn Long Context | Relevance Detection | Irrelevance Detection |
|-------------|-----------------------------|------------------|---------------------|-----------------------|-----------------------|-------------------------------|------------------|---------------------|-----------------------|-----------------------|-------------------------------|---------|----------------|------------------|------------------|--------------------------|---------------|----------------|--------------------|--------------------|-----------------------|------------------|------------------|
| 52.21%      | hqfx/octupus-tool-call-v1   | 80.69%           | 65.25%              | 94.00%               | 82.50%               | 81.00%                         | 83.70%           | 88.29%              | 92.00%               | 82.00%               | 72.50%                         | 73.92%  | 66.28%         | 69.72%           | 31.25%           | 33.33%                  | 0.12%         | 0.00%          | 0.00%              | 0.50%              | 0.00%                | 73.17%           | 84.46%           |
| 55.69%      | hqfx/octupus-tool-call-v2     | 84.65%           | 69.08%              | 93.00%               | 90.00%               | 86.50%                         | 78.48%           | 83.43%              | 86.00%               | 82.00%               | 62.50%                         | 79.52%  | 71.32%         | 73.88%           | 62.50%           | 66.67%                  | 4.50%         | 8.00%          | 1.00%              | 6.50%              | 2.50%                | 66.67%           | 92.28%           |


在[BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html)上面可以排到25名，multi_turn有了一些提升。