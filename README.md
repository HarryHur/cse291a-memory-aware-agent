# Memory-Aware Agent Edge Serving

**UC San Diego - CSE 291A (Fall 2026), Phase 1**  
**Author:** Jungwook Hur

## Overview

This project studies how the number of concurrently running AI agent tasks affects local LLM serving performance on a resource-limited device. An agent task makes two LLM calls with a simulated tool step between them. We compare workloads with different input context lengths and generation limits.

**Phase 1 goal:** Build a reproducible fixed-concurrency baseline, measure task throughput, latency, queueing, and process memory, and use the results to motivate a focused scheduling policy for Phase 2. Phase 1 does **not** implement an adaptive scheduler.

## Repository contents

```text
.
|-- README.md
|-- workload.py                 # Agent workload generator and measurement code
|-- analyze_ph1.py             # Analysis and plotting script
|-- results/
|   |-- run_summaries.csv       # One row per experimental run (36 rows)
|   |-- task_results.csv        # One row per agent task (216 rows)
|   `-- request_results.csv     # One row per LLM call (432 rows)
`-- plots/                      # Generated Phase 1 plots (by analyze_ph1.py)
    |-- throughput_vs_concurrency.png
    |-- request_latency_vs_concurrency.png
    |-- task_queue_vs_concurrency.png
    |-- memory_vs_context.png
    |-- throughput_by_output_size_vs_concurrency.png  # Additional view
    |-- throughput_vs_output_size.png
    `-- request_latency_vs_output_size.png
```

The three CSVs under `results/` are the final, cleaned results used for the Phase 1 report. Two invalid measurement attempts were excluded and rerun (see **Run protocol** below). The repository does not need to include the downloaded model weights.

## Test environment

| Item | Configuration |
|---|---|
| Device | MacBook Pro with Apple M1 Pro (10-core CPU), 32 GB unified memory |
| OS | macOS, ARM64 |
| Inference server | `llama.cpp` v0.6.0, build 11429, commit `d81235049` |
| Model | `Qwen/Qwen3-4B-GGUF:Q4_K_M` |
| Inference API | Local OpenAI-compatible endpoint, `http://127.0.0.1:8080/v1/chat/completions` |
| Model settings | Temperature 0; reasoning disabled |
| Server context | 40,960 tokens in the measured environment |
| Server parallel slots | 4 in the measured environment |
| Python workload dependencies | `requests`, `psutil` for the workload; `pandas`, `matplotlib` for analysis |

Performance numbers depend on the hardware, model, server version, and runtime configuration. Check the server startup log for the effective context size and number of slots when reproducing the experiment.

## Setup

1. Install a compatible `llama.cpp` build that provides `llama-server`. The original experiment used the version listed above.
2. Install Python 3 and the workload dependencies:

   ```bash
   python3 -m pip install requests psutil pandas matplotlib
   ```

3. Start the model server in one terminal:

   ```bash
   llama-server -hf Qwen/Qwen3-4B-GGUF:Q4_K_M --reasoning off
   ```

4. Wait for model initialization and the server's built-in warm-up to finish before starting a measured run. The server must remain running while `workload.py` executes.

The model is obtained through the `-hf` option; its download may be required on the first startup. The project does not make calls to a hosted LLM API.

## Agent workload

Each agent task performs the following steps:

```text
LLM call 1 -> simulated tool delay (1 second) -> LLM call 2 -> task complete
```

- **LLM call 1:** Reads a synthetic input context, finds a request-specific experiment code, and is instructed to repeat the word `TOKEN` until the generation limit is reached.
- **Tool step:** Sleeps for one second and produces a synthetic `VERIFIED` result. No external tool API is called.
- **LLM call 2:** Receives the original context, the first LLM output, the tool result, and a final-answer instruction. This makes the second prompt longer and represents context growth between agent steps.
- **Concurrency:** A Python `ThreadPoolExecutor(max_workers=C)` limits the number of simultaneous *agent tasks*, not tokens or individual model requests. All tasks are submitted in order.

The **same `max_tokens` limit applies to both LLM calls**. In the recorded experiments, the first call generated exactly 32 or 192 tokens, depending on the condition, while the second call produced a short final answer. The output-length comparison therefore includes both the extra first-call generation work and its effect on the next call's input length; it is not a decode-only microbenchmark.

## Experimental design

### Controlled variables

| Variable | Values |
|---|---|
| Initial context size | `short` (~193 prompt tokens), `medium` (~2,181), `long` (~8,481) for the first LLM call |
| Output condition | `short` (`max_tokens=32`), `long` (`max_tokens=192`) |
| Maximum concurrent agent tasks | `1`, `2`, `4` |
| Agent tasks per run | `6` |
| Repetitions per condition | `3` |
| Simulated tool delay | `1.0` second |

The context-size values describe synthetic *input lengths*. The model server context-window setting stays the same across experimental conditions. The prompt-token counts above come from the recorded first-call request results.

### Experiment matrix

- **Context sweep:** Short output at short/medium/long context, each at concurrency 1/2/4 = **9 conditions**.
- **Output sweep:** Medium context with long output at concurrency 1/2/4 = **3 additional conditions**. Compare these against the medium-context/short-output runs in the first sweep.

**Total: 12 conditions x 3 repetitions = 36 valid runs, with 216 agent tasks and 432 LLM requests.**

## How to reproduce one run

The command syntax is:

```bash
python3 workload.py <concurrency> <context_size> <output_size> [num_tasks] [repetition]
```

For example, the first repetition of short context, short output, and concurrency 1 is:

```bash
python3 workload.py 1 short short 6 1
```

Another example, medium context with long output and concurrency 4:

```bash
python3 workload.py 4 medium long 6 1
```

**Important:** Pass `6` explicitly as `num_tasks`. The original script has a default of 8 tasks if this argument is omitted, but all formal Phase 1 measurements used 6 tasks.

To reproduce the full matrix, run each of the 12 conditions three times using repetition IDs `1`, `2`, and `3`. **Manually stop and restart `llama-server` before every run**, wait for initialization and built-in warm-up, then execute exactly one workload command. This was also the procedure used to collect the original data; the Python script itself does not restart the server.

Results are **appended** to CSV files under `results/`. To preserve the submitted data, perform new experiments from a separate working directory or back up the existing CSVs first.

### Run protocol and exclusions

Each measured run began with a fresh server process to avoid carrying server prompt-cache state across runs. During data collection, one long-context/C2 attempt exceeded the original 60-second client timeout. The request timeout was raised to 300 seconds so that valid long requests could finish, and the condition was rerun. A separate long-context/C1 attempt was affected by unrelated background application activity and was also rerun. Neither invalid attempt is included in the 36 reported runs. The final dataset contains no failed or timed-out requests.

## Recorded metrics and data format

All CSVs contain `run_id`, the experimental condition, and a repetition number for joining records.

- **`request_results.csv`:** Each LLM call's status, request duration (`end_to_end_latency`), server-reported prefill/decode timings, prompt/completion/cache token counts, and observed client queue time. The measured request duration includes any waiting inside the inference server; it is **not** isolated GPU execution time.
- **`task_results.csv`:** Each agent task's status, time waiting for a worker (`task_queue_time`), execution time, full submission-to-completion latency (`task_end_to_end_time`), tool delay, and associated request IDs.
- **`run_summaries.csv`:** Run duration, throughput (completed tasks or requests divided by run duration), mean/p50/p95 latencies, queue times, completion/failure counts, and process memory measurements.

Task p95 uses the nearest-rank method. Since there are only six tasks in a run, the within-run task p95 is close to the maximum and should be interpreted cautiously.

Memory is measured by sampling `llama-server` process RSS approximately every 0.1 seconds. The values are converted using `1024 ** 3`, so their correct physical unit is **GiB**. The original CSV column names retain the historical `_gb` suffix (`idle_rss_gb`, `peak_rss_gb`, `peak_increase_gb`) for compatibility, but their numerical values represent GiB. Process RSS is a memory-pressure indicator, **not** dedicated GPU VRAM usage.

## Analysis and main findings

The included `analyze_ph1.py` script is used to summarize the final CSV data and generate the Phase 1 plots:

```bash
python3 analyze_ph1.py
```

The analysis requires `pandas` and `matplotlib` (included in the setup command above). It writes seven plots to `plots/`; four of these are used in the report, while the other plots provide additional views of the same experiment results.

The main observations are:

1. **Short contexts benefit from concurrency:** Mean completed-task throughput increases from about **0.297 tasks/s (C1)** to **0.465 tasks/s (C4)**.
2. **Long contexts do not scale in throughput:** Throughput remains near **0.015 tasks/s**, while average LLM request latency increases from **31.87 s (C1)** to **117.51 s (C4)**.
3. **Longer inputs increase memory pressure:** At C1 with short output, mean peak server RSS increases from **8.44 GiB (short context)** to **13.90 GiB (long context)**.
4. **Longer first-step outputs add cost:** At medium context, the 192-token output condition lowers completed-task throughput compared with the 32-token condition at every tested concurrency.

These measurements suggest that heavy agent workloads compete for limited inference resources. They do **not** establish whether compute capacity, memory bandwidth, or server-side scheduling is the specific hardware-level bottleneck.

## Limitations and next step

Phase 1 uses one device, one quantized model, synthetic contexts, a fixed tool delay, and homogeneous workloads per run. It does not measure separate tool-idle durations or independently isolate retained-prefix cache effects. Phase 2 will evaluate a simple workload- or memory-pressure-aware admission policy against fixed-concurrency baselines using mixed workloads and changing task arrival rates.
