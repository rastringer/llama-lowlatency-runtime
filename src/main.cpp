// Minimal low-latency inference runtime built on top of libllama.

// Goals:
// - Stream generated tokens
// - Measure time-to-first-token (TTFT)
// - Measure steady-state inter-token latency
// - Build a small benchmark harness for CPU inference behaviour

// Links directly to libllama C-style API to expose inference without subprocess execution of llama-cli.

#include "llama.h"

#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

// Monotonic clock to avoid wall clock adjustments (for timezone changes etc)
using Clock = std::chrono::steady_clock;

// Latency percentiles (p50/p95/p99) for inter-token timings
// Percentiles are important since tail latency can dominate user experience (vs averages)
static double percentile(std::vector<double> values, double p) {
    if (values.empty()) {
        return 0.0;
    }

    std::sort(values.begin(), values.end());

    const double idx = (p / 100.0) * (values.size() - 1);
    const size_t lo = static_cast<size_t>(idx);
    const size_t hi = std::min(lo+1, values.size() - 1);

    const double frac = idx - lo;
    return values[lo] * (1.0 - frac) + values[hi] * frac;
}

// Convert chrono timestamps -> milliseconds
static double ms_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// Supress llama.cpp logs to keep output focused on latency metrics
static void llama_log_callback(ggml_log_level level, const char* text, void* user_data) {
    (void) level;
    (void) text;
    (void) user_data;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./llama_latency <model.gguf> <prompt>\n";
        return 1;
    }

const std::string model_path = argv[1];
const std::string prompt = argv[2];

llama_log_set(llama_log_callback, nullptr);

// Initialize llama backend before loading model
llama_backend_init();

// Model loading -- quantized GGUF model -> memory
// Excluded from TTFT since serving systems typically load models 
// once and reuse
llama_model_params model_params = llama_model_default_params();
llama_model* model = llama_model_load_from_file(model_path.c_str(), model_params);

if (!model) {
    std::cerr << "Failed to load model\n";
    return 1;
}

// Inference context
// n_ctx sets max content window
// n_batch sets prompt processing throughput
llama_context_params ctx_params = llama_context_default_params();
ctx_params.n_ctx = 2048;
ctx_params.n_batch = 512;

llama_context* ctx = llama_init_from_model(model, ctx_params);

if (!ctx) {
    std::cerr << "Failed to create context\n";
    llama_model_free(model);
    return 1;
}

const llama_vocab* vocab = llama_model_get_vocab(model);

// Convert prompt into tokens. Tokenization cost measured separately 
// from inference latency
const int n_prompt = -llama_tokenize(
    vocab,
    prompt.c_str(),
    prompt.size(),
    nullptr,
    0,
    true,
    true
);

std::vector<llama_token> prompt_tokens(n_prompt);

llama_tokenize(
    vocab,
    prompt.c_str(),
    prompt.size(),
    prompt_tokens.data(),
    prompt_tokens.size(),
    true,
    true
);

auto t_start = Clock::now();

llama_batch batch = llama_batch_get_one(
    prompt_tokens.data(),
    prompt_tokens.size()
);

if (llama_decode(ctx, batch) != 0) {
    std::cerr << "Prompt decode failed\n";
    return 1;
}

// Prefil phase: run prompt through transformer once to 
// populate KV cache before autoregressive generation begins
// Large prompts affect prefil latency
auto t_after_prefill = Clock::now();

llama_token new_token_id;
int generated = 0;
constexpr int max_tokens = 42;

std::vector<double> inter_token_ms;
Clock::time_point first_token_time;
Clock::time_point prev_token_time;

// Greedy sampler -- avoid temperature / sampling variability while
// validating latency
auto sparams = llama_sampler_chain_default_params();
llama_sampler* sampler = llama_sampler_chain_init(sparams);
llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

// Autoregressive decode: 
// Sample next token from logits -> emit token (streaming) 
// -> feed token back into model -> repeat
for (int pos = n_prompt; pos < n_prompt + max_tokens; ++pos) {
    new_token_id = llama_sampler_sample(
        sampler,
        ctx,
        -1
    );

    if (llama_vocab_is_eog(vocab, new_token_id)) {
        break;
    }

    char buf[256];
    int n = llama_token_to_piece(
        vocab,
        new_token_id,
        buf,
        sizeof(buf),
        0,
        true
    );

    auto now = Clock::now();

    if (generated == 0) {
        first_token_time = now;
    } else {
        inter_token_ms.push_back(ms_between(prev_token_time, now));
    }

    prev_token_time = now;

    if (n > 0) {
        std::cout << std::string(buf, n) << std::flush;
    }

    llama_token token = new_token_id;
    batch = llama_batch_get_one(&token, 1);

    if (llama_decode(ctx, batch) != 0) {
        std::cerr << "\nDecode failed\n";
        break;
    }
    generated++;
}

// Measure latency between emitted tokens
// TTFT caputures startup cost, inter-token latency captures 
// sustained throughput 
auto t_end = Clock::now();

double ttft_ms = generated > 0 ? ms_between(t_start, first_token_time) : 0.0;

double mean_inter_token = 0.0;
for (double x : inter_token_ms) {
    mean_inter_token += x;
}
if (!inter_token_ms.empty()) {
    mean_inter_token /= inter_token_ms.size();
}

double p50_inter_token = percentile(inter_token_ms, 50.0);
double p95_inter_token = percentile(inter_token_ms, 95.0);
double p99_inter_token = percentile(inter_token_ms, 99.0);

std::cerr << "\n\n=== Latency Metrics ===\n";
std::cerr << "prompt_tokens: " << n_prompt << "\n";
std::cerr << "generated_tokens: " << generated << "\n"; 
std::cerr << "prefill_ms: " << ms_between(t_start, t_after_prefill) << "\n";
std::cerr << "ttft_ms: " << ttft_ms << "\n";
std::cerr << "mean_inter_token_ms: " << mean_inter_token << "\n";
std::cerr << "total_ms: " << ms_between(t_start, t_end) << "\n";
std::cerr << "p50_inter_token_ms: " << p50_inter_token << "\n";
std::cerr << "p95_inter_token_ms: " << p95_inter_token << "\n";
std::cerr << "p99_inter_token_ms: " << p99_inter_token << "\n";

llama_sampler_free(sampler);
llama_free(ctx);
llama_model_free(model);
llama_backend_free();

return 0;

}
