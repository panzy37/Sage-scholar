# Remote PDF image embeddings

PDF parsing, text E5/BGE retrieval, Evidence and indexes remain on the application host.
Only image pixels and visual-search query text go to the configured vLLM service.

## Server

Use the existing `qwen-vl-embed` environment and downloaded model directory.
The verified server for this integration runs vLLM 0.33.0 with PyTorch 2.14.0+cu130.

```bash
conda activate qwen-vl-embed
CUDA_VISIBLE_DEVICES=0 vllm serve "$HOME/models/Qwen3-VL-Embedding-2B" \
  --runner pooling --served-model-name qwen3-vl-embedding-2b \
  --dtype bfloat16 --max-model-len 8192 --max-num-seqs 4 \
  --gpu-memory-utilization 0.35 --host 127.0.0.1 --port 18081
```

Use an SSH tunnel (replace connection details); no public inference port is required:

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L 18001:127.0.0.1:18081 user@server
curl --fail http://127.0.0.1:18001/v1/models
```

For Docker, the backend needs a reachable tunnel endpoint; its `127.0.0.1` is not
the Mac host. This setup is for the native Mac backend. Do not expose an unauthenticated
forward publicly to accommodate Docker.

## Application configuration

```dotenv
VISUAL_EMBEDDING_PROVIDER=remote
VISUAL_EMBEDDING_BASE_URL=http://127.0.0.1:18001/v1
VISUAL_EMBEDDING_MODEL=qwen3-vl-embedding-2b
VISUAL_EMBEDDING_REVISION=9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda-vllm0.33.0
VISUAL_EMBEDDING_DIMENSIONS=2048
VISUAL_EMBEDDING_TIMEOUT=120
VISUAL_EMBEDDING_API_KEY=
```

Restart the application after changing configuration. API keys are read from the
environment at request time and never included in encoder identity.
The client bypasses global proxies for this explicit service URL and sends local images
as data URLs. It uses Qwen's system instruction, empty assistant prefix,
`continue_final_message=true` and `add_special_tokens=true` for both modalities.
Requests are bounded by HTTP timeouts, checked for cancellation before/after each item,
and never silently retried or replaced with unrelated text embeddings.

The revision is operator-declared: pin the deployed weights/backend and update it on
server changes. The client cannot infer an immutable remote weight revision from the
OpenAI model listing. vLLM preprocessing can differ from sentence-transformers;
the identity includes the backend/template, so existing local visual indexes need
rebuilding. The text index remains unchanged. Remote failures preserve the existing
text-only warning/fallback behavior.
