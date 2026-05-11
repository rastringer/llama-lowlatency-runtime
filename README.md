# Low-latency C++ LLM inference 

This repo is a work-in-progress for a low-latency C++ LLM inference harness and serving runtime on top of llama.cpp.


## Build steps

Clone the repo and build:

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

To run:

```
./build/llama_latency ./models/<your-model>.gguf "Hello, my name is..."
```

Coming updates will cover how best to import `GGUF` models and other additions.

