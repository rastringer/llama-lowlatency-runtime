# Low-latency C++ LLM inference 

Low-latency C++ inference runtime built on top of llama.cpp, focused on streaming token generation and latency analysis (TTFT, inter-token latency). 

## Features

- Direct libllama integration without subprocess calls
- Streaming token output
- Time-to-first-token (TTFT) measurement
- Prefill vs decode latency breakdown
- Inter-token latency tracking
- Simple, minimal C++20 implementation

## Build steps

Clone the repo and build:

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

To run, you'll first need to download a GGUF model. Some good and small examples to experiment with include Phi2, Qwen0.5b and TinyLlama.

```
mkdir -p models
cd models

wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -O tinyllama.gguf
```

## Run

```
./build/llama_latency ./models/<your-model>.gguf "Hello, my name is..."
```

Coming updates will cover how best to import `GGUF` models and other additions.

## Example Output

```
"Hello, my name is Mustafa and I work as..." 
 [insert job title here] I am a [insert job title here] and I am passionate about [insert job title here] and I am always looking for ways to improve my skills and knowledge. I am

=== Latency Metrics ===
prompt_tokens: 12
generated_tokens: 42
prefill_ms: 32.6847
ttft_ms: 33.396
mean_inter_token_ms: 9.56672
total_ms: 434.562
p50_inter_token_ms: 9.29288
p95_inter_token_ms: 11.1385
p99_inter_token_ms: 11.3254
```