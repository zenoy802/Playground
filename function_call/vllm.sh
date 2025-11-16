# change the path to your specific models from HF or other sources
# vllm serve /root/autodl-tmp/.cache/hub/models--Qwen--QwQ-32B/snapshots/976055f8c83f394f35dbd3ab09a285a984907bd0/ \
# vllm serve /root/autodl-tmp/.cache/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2/ \
vllm serve /root/autodl-fs/models/Qwen/QwQ-32B/ \
--gpu-memory-utilization 0.95 \
--max-model-len 1024 \
--port 8000 \
--host 0.0.0.0 \
--enable-log-requests \
--enable-log-outputs \
--reasoning-parser deepseek_r1 \
--enable-auto-tool-choice \
--tool-call-parser hermes \