# Description
Optuna made with the following command

llama-server \
    -hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_XL \
    --ctx-size 51000 \
    --threads -1 \
    --temp 0.7 \
    --min-p 0.0 \
    --top-p 0.80 \
    --top-k 20 \
    --presence-penalty 1.0 \
    --flash-attn on &