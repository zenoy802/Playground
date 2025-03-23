# change the path to your specific models from HF or other sources
vllm serve /root/autodl-tmp/.cache/hub/models--Qwen--QwQ-32B/snapshots/976055f8c83f394f35dbd3ab09a285a984907bd0/ \
--gpu-memory-utilization 0.95 \
--max-model-len 2048 \
--port 8000 \
--host 0.0.0.0 \
--reasoning-parser deepseek_r1 \
--enable-auto-tool-choice \
--tool-call-parser hermes