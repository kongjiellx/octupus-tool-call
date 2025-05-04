VLLM_USE_V1=0 python qwen_server.py \
    --model Qwen/Qwen3-32B \
    --tensor-parallel-size 2 \
    --max-model-len 32768
