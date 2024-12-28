# octupus-tool-call 

[English|[中文](https://github.com/kongjiellx/octupus-tool-call/blob/main/README.md)]

## Project Overview

This project focuses on training models for tool calling (function calling). It aims to make all data, code, models, and inference methods used in the training process open source.

## Project Goals & Evaluation Metrics

Currently, the project uses the [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) as the evaluation benchmark. The goal is to maximize the model's performance on BFCL. Continuous iterations will be made, and improvements from each version will be shared here.

## Function Calling

Function calling refers to a method that enables LLMs to output function call instructions. While LLMs are trained on diverse text formats and tend to respond to users in natural language, their ability to output well-structured function call instructions is generally weaker.

This project optimizes this issue in two ways:
1. Training LLMs using data specifically formatted for function calling.
2. Enforcing format constraints during inference.

## Key Points of the Approach

1. **Base Model**: Using the Qwen-2.5 series pre-trained model for instruction fine-tuning with open-source dialogue data and function calling data.  
   - Instruction-tuned models are not used because the chat template needs to be altered. To enable format constraints during inference, special tokens are utilized.
2. **Output Formatting**: Leveraging [lm-format-enforcer](https://github.com/noamgat/lm-format-enforcer) to ensure the generated function call instructions adhere to valid formats.
3. **Inference Server**: A server compatible with the OpenAI API that supports tool selection, parallel tool calls, and ensures the validity of output.

## Directory Overview

- `utils/edit_tokenizer_and_model.py`: Edits the tokenizer to add special tokens and initialize them accordingly.
- `run`: Model training scripts
  - `train_stg1.sh`: Trains on chat-format data only.
  - `train_stg2.sh`: Continues training from stage 1 with mixed chat and function call data.
- `inference`: OpenAI-compatible model service using structured outputs.
  - Usage:  
    ```bash
    python oai_server.py --model /your/model --tensor-parallel-size 1 --max-model-len 8192
    ```

## Evaluation Process

1. Start the model service using `oai_server.py`.
2. Run BFCL directly for testing without modifying the service. Fill in any supported GPT function-calling model name to trigger the OpenAI FC model logic in BFCL.
   ```bash
   OPENAI_BASE_URL=http://localhost:8000 bfcl generate --model gpt-4-turbo-2024-04-09-FC --num-threads 8 --test-category all
   ```

---

## Iteration Log

### V1

#### Data
- **Stage 1**:
  - Re-answered [open-hermes](https://huggingface.co/datasets/teknium/OpenHermes-2.5) using Qwen2.5-72b-instruct.
  - [Wild-Chat](https://huggingface.co/datasets/allenai/WildChat) bilingual data.
- **Stage 2**:
  - [hermes-function-calling](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1).
  - [Custom Chinese long function-calling data](https://huggingface.co/datasets/hqfx/fc_zh_hard).
  - Function call data underwent filtering; see the `data` directory for specific logic.

#### Model and Data Repository
[https://huggingface.co/collections/hqfx/hqfx-octupus-tool-call-v1-6752bc1b3d5dc4e06f394e59](https://huggingface.co/collections/hqfx/hqfx-octupus-tool-call-v1-6752bc1b3d5dc4e06f394e59)

#### Evaluation Results

| Overall Acc | Model                        | Non-Live AST Acc | Non-Live Simple AST | Non-Live Multiple AST | Non-Live Parallel AST | Non-Live Parallel Multiple AST | Non-Live Exec Acc | Non-Live Simple Exec | Non-Live Multiple Exec | Non-Live Parallel Exec | Non-Live Parallel Multiple Exec | Live Acc | Live Simple AST | Live Multiple AST | Live Parallel AST | Live Parallel Multiple AST | Multi Turn Acc | Multi Turn Base | Multi Turn Miss Func | Multi Turn Miss Param | Multi Turn Long Context | Relevance Detection | Irrelevance Detection |
|-------------|-----------------------------|------------------|---------------------|-----------------------|-----------------------|-------------------------------|------------------|---------------------|-----------------------|-----------------------|-------------------------------|---------|----------------|------------------|------------------|--------------------------|---------------|----------------|--------------------|--------------------|-----------------------|------------------|------------------|
| 52.21%      | hqfx/octupus-tool-call-v1   | 80.69%           | 65.25%              | 94.00%               | 82.50%               | 81.00%                         | 83.70%           | 88.29%              | 92.00%               | 82.00%               | 72.50%                         | 73.92%  | 66.28%         | 69.72%           | 31.25%           | 33.33%                  | 0.12%         | 0.00%          | 0.00%              | 0.50%              | 0.00%                | 73.17%           | 84.46%           |

- **BFCL Rank**: Top 30 (slightly surpassing GPT-3.5).
- **Multi-Turn Score**: ~0%, pending further analysis.

---

### V2

#### Data
- Added cleaned [ToolAce](https://huggingface.co/datasets/Team-ACE/ToolACE) dataset (cleaning scripts in `data` directory).
- Reformatted all data to OpenAI’s tool-calling format, adapting training and inference accordingly.

#### Inference Service
- Simplified code and fixed bugs (e.g., supporting cases where function names are prefixes of one another).

#### Model and Data Repository
https://huggingface.co/collections/hqfx/octupus-tool-call-v2-676fa1c11c48eff17fa1c017

#### Evaluation Results

| Overall Acc | Model                        | Non-Live AST Acc | Non-Live Simple AST | Non-Live Multiple AST | Non-Live Parallel AST | Non-Live Parallel Multiple AST | Non-Live Exec Acc | Non-Live Simple Exec | Non-Live Multiple Exec | Non-Live Parallel Exec | Non-Live Parallel Multiple Exec | Live Acc | Live Simple AST | Live Multiple AST | Live Parallel AST | Live Parallel Multiple AST | Multi Turn Acc | Multi Turn Base | Multi Turn Miss Func | Multi Turn Miss Param | Multi Turn Long Context | Relevance Detection | Irrelevance Detection |
|-------------|-----------------------------|------------------|---------------------|-----------------------|-----------------------|-------------------------------|------------------|---------------------|-----------------------|-----------------------|-------------------------------|---------|----------------|------------------|------------------|--------------------------|---------------|----------------|--------------------|--------------------|-----------------------|------------------|------------------|
| 52.21%      | hqfx/octupus-tool-call-v1   | 80.69%           | 65.25%              | 94.00%               | 82.50%               | 81.00%                         | 83.70%           | 88.29%              | 92.00%               | 82.00%               | 72.50%                         | 73.92%  | 66.28%         | 69.72%           | 31.25%           | 33.33%                  | 0.12%         | 0.00%          | 0.00%              | 0.50%              | 0.00%                | 73.17%           | 84.46%           |
| 55.69%      | hqfx/octupus-tool-call-v2   | 84.65%           | 69.08%              | 93.00%               | 90.00%               | 86.50%                         | 78.48%           | 83.43%              | 86.00%               | 82.00%               | 62.50%                         | 79.52%  | 71.32%         | 73.88%           | 62.50%           | 66.67%                  | 4.50%         | 8.00%          | 1.00%              | 6.50%              | 2.50%                | 66.67%           | 92.28%           |

- **BFCL Rank**: Top 25.
- **Improvements**: Significant gains in multi-turn capabilities and overall accuracy.

---