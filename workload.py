import time
import requests
import sys
import threading
import psutil
import csv
import os
import statistics
import math
from datetime import datetime

from concurrent.futures import ThreadPoolExecutor, as_completed


URL = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "Qwen/Qwen3-4B-GGUF:Q4_K_M"

if len(sys.argv) not in {4, 5, 6}:
    print(
        "Usage: python workload.py "
        "<concurrency> <context_size> <output_size> [num_tasks] [repetition]"
    )
    sys.exit(1)

CONCURRENCY = int(sys.argv[1])
CONTEXT_SIZE = sys.argv[2].lower()
OUTPUT_SIZE = sys.argv[3].lower()
NUM_TASKS = int(sys.argv[4]) if len(sys.argv) >= 5 else 8
REPETITION = int(sys.argv[5]) if len(sys.argv) == 6 else 0

if CONTEXT_SIZE not in {"short", "medium", "long"}:
    print("Error: context_size must be short, medium, or long.")
    sys.exit(1)

if OUTPUT_SIZE not in {"short", "long"}:
    print("Error: output_size must be short or long")
    sys.exit(1)

if OUTPUT_SIZE == "short":
    MAX_TOKENS = 32
else:
    MAX_TOKENS = 192

BASE_CONTEXT = (
    "The serving system receives requests, processes input prompts, "
    "generates output tokens, and records latency, queueing delay, "
    "throughput, and memory usage. "
)

TARGET_WORDS = {
    "short": 80,
    "medium": 1500,
    "long": 6000,
}

TOOL_DELAY_SECONDS = 1.0

# Find llama-server process for memory monitoring
def find_llama_server():
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = proc.info["name"] or ""
            cmdline = " ".join(proc.info["cmdline"] or [])

            if "llama-server" in name or "llama-server" in cmdline:
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return None

# Monitor llama-server memory usage while workload is running
def monitor_memory(proc, stop_event, samples):
    while not stop_event.is_set():
        try:
            samples.append(proc.memory_info().rss)
        except psutil.NoSuchProcess:
            break

        stop_event.wait(0.1)

# Send one request to the LLM and collect request metrics
def send_request(request_id, prompt, arrival_time):
    service_start = time.perf_counter()

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0,
        "max_tokens": MAX_TOKENS
    }

    try:
        response = requests.post(
            URL,
            json=payload,
            timeout=300
        )

        end_time = time.perf_counter()

        response.raise_for_status()
        data = response.json()

        usage = data["usage"]
        timings = data["timings"]

        return {
            "request_id": request_id,
            "prompt": prompt,
            "answer": data["choices"][0]["message"]["content"],

            "status": "success",
            "error": None,

            "queue_time": service_start - arrival_time,
            "service_time": end_time - service_start,
            "end_to_end_latency": end_time - arrival_time,

            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "cached_tokens": usage["prompt_tokens_details"]["cached_tokens"],

            "server_prefill_ms": timings["prompt_ms"],
            "server_decode_ms": timings["predicted_ms"],
            "decode_tps": timings["predicted_per_second"]
        }

    except requests.Timeout:
        end_time = time.perf_counter()

        return {
            "request_id": request_id,
            "prompt": prompt,
            "answer": None,

            "status": "timeout",
            "error": "Request timed out",

            "queue_time": service_start - arrival_time,
            "service_time": end_time - service_start,
            "end_to_end_latency": end_time - arrival_time,

            "prompt_tokens": None,
            "completion_tokens": None,
            "cached_tokens": None,
            "server_prefill_ms": None,
            "server_decode_ms": None,
            "decode_tps": None
        }

    except requests.RequestException as e:
        end_time = time.perf_counter()

        return {
            "request_id": request_id,
            "prompt": prompt,
            "answer": None,

            "status": "failure",
            "error": str(e),

            "queue_time": service_start - arrival_time,
            "service_time": end_time - service_start,
            "end_to_end_latency": end_time - arrival_time,

            "prompt_tokens": None,
            "completion_tokens": None,
            "cached_tokens": None,
            "server_prefill_ms": None,
            "server_decode_ms": None,
            "decode_tps": None
        }

# Run one agent task: LLM -> tool wait -> LLM
def run_agent_task(task_id, context, first_prompt, task_arrival_time):
    task_start = time.perf_counter()

    # Step 1: first LLM call
    first_arrival_time = time.perf_counter()
    first_result = send_request(
        task_id * 2 - 1, #2 LLM requests per agent task
        first_prompt,
        first_arrival_time
    )
    first_result["task_id"] = task_id
    first_result["agent_step"] = "llm_1"

    # Stop the task if the first LLM call failed
    if first_result["status"] != "success":
        task_end = time.perf_counter()
        return {
            "task_id" : task_id,
            "status" : "failure",
            "task_queue_time": task_start - task_arrival_time,
            "task_end_to_end_time": task_end - task_arrival_time,
            "task_time" : task_end - task_start,
            "tool_delay" : 0.0,
            "first_result" : first_result,
            "second_result" : None,
        }

    # Step 2: simulated tool execution
    time.sleep(TOOL_DELAY_SECONDS)

    tool_result = f"Tool result for task {task_id}: VERIFIED"

    # Step 3: second LLM call
    second_prompt = (
        f"Original context:\n{context}\n\n"
        f"First model output:\n{first_result['answer']}\n\n"
        f"Tool result:\n{tool_result}\n\n"
        "Using the first model output and the tool result, "
        "produce the final answer for this task."
        )
    second_arrival_time = time.perf_counter()

    second_result = send_request(
        task_id * 2,
        second_prompt,
        second_arrival_time
    )

    second_result["task_id"] = task_id
    second_result["agent_step"] = "llm_2"

    task_end = time.perf_counter()

    return {
        "task_id": task_id,
        "status": (
            "success" if second_result["status"] == "success"
            else "failure"
        ),
        # Time waiting for a worker
        "task_queue_time": task_start - task_arrival_time,
        #From submit to end (queue + task_time)
        "task_end_to_end_time": task_end - task_arrival_time,
        #Time after agent task started (LLM #1 + tool wait + LLM #2)
        "task_time": task_end - task_start,
        "tool_delay": TOOL_DELAY_SECONDS,
        "first_result": first_result,
        "second_result": second_result,
    }

# Print result for one LLM request
def print_request_result(result):
    print(
        f"\n--- Request {result['request_id']} "
        f"| Task {result['task_id']} "
        f"| {result['agent_step']} ---"
    )

    print(f"Status: {result['status']}")

    print(
        f"Latency: "
        f"queue={result['queue_time']:.3f}s | "
        f"service={result['service_time']:.3f}s | "
        f"E2E={result['end_to_end_latency']:.3f}s"
    )

    if result["status"] != "success":
        print(f"Error: {result['error']}")
        return

    print(
        f"Tokens: "
        f"prompt={result['prompt_tokens']} | "
        f"completion={result['completion_tokens']} | "
        f"cached={result['cached_tokens']}"
    )

    print(
        f"Server: "
        f"prefill={result['server_prefill_ms']:.2f}ms | "
        f"decode={result['server_decode_ms']:.2f}ms | "
        f"{result['decode_tps']:.2f} tok/s"
    )

    answer_preview = result["answer"].replace("\n", " ")[:100]
    print(f"Answer: {answer_preview}...")

# Print result for one complete agent task
def print_task_result(task_result):
    print(f"\n=== Agent Task {task_result['task_id']} ===")

    print(
        f"Status: {task_result['status']} | "
        f"Queue: {task_result['task_queue_time']:.3f}s | "
        f"Tool: {task_result['tool_delay']:.3f}s"
    )

    print(
        f"Time: "
        f"execution={task_result['task_time']:.3f}s | "
        f"E2E={task_result['task_end_to_end_time']:.3f}s"
    )

    first_result = task_result["first_result"]
    second_result = task_result["second_result"]

    first_request_id = (
        first_result["request_id"]
        if first_result is not None
        else "N/A"
    )

    second_request_id = (
        second_result["request_id"]
        if second_result is not None
        else "N/A"
    )

    print(
        f"Requests: "
        f"llm_1={first_request_id} | "
        f"llm_2={second_request_id}"
    )

# Print overall workload summary
def print_run_summary(
    run_id,
    total_time,
    request_summary,
    task_summary,
    idle_rss_gb,
    peak_rss_gb,
    peak_increase_gb
):
    print("\n=== Workload Summary ===")

    print(
        f"Config: concurrency={CONCURRENCY} | "
        f"tasks={NUM_TASKS} | "
        f"total={total_time:.3f}s"
    )

    print(
        f"Requests: "
        f"completed={request_summary['completed_count']} | "
        f"failed={request_summary['failed_count']} | "
        f"timeout={request_summary['timeout_count']} | "
        f"throughput={request_summary['throughput']:.3f} req/s"
    )

    print(
        f"Request latency: "
        f"avg={format_metric(request_summary['average_latency'])}s | "
        f"p50={format_metric(request_summary['p50_latency'])}s | "
        f"p95={format_metric(request_summary['p95_latency'])}s | "
        f"queue={format_metric(request_summary['average_queue_time'])}s"
    )

    print(
        f"Tasks: "
        f"completed={task_summary['completed_count']} | "
        f"failed={task_summary['failed_count']} | "
        f"throughput={task_summary['throughput']:.3f} task/s"
    )

    print(
        f"Task latency: "
        f"avg={format_metric(task_summary['average_latency'])}s | "
        f"p50={format_metric(task_summary['p50_latency'])}s | "
        f"p95={format_metric(task_summary['p95_latency'])}s | "
        f"queue={format_metric(task_summary['average_queue_time'])}s"
    )

    print(
        f"Memory: "
        f"idle={idle_rss_gb:.3f} GB | "
        f"peak={peak_rss_gb:.3f} GB | "
        f"increase={peak_increase_gb:.3f} GB"
    )

    print(f"Run ID: {run_id}")

# Save request-level results
def save_request_results(request_results, run_id):
    os.makedirs("results", exist_ok=True)

    request_file = "results/request_results.csv"

    request_fields = [
        "run_id",
        "context_size",
        "output_size",
        "concurrency",
        "num_tasks",
        "repetition",
        "task_id",
        "agent_step",
        "request_id",
        "status",
        "error",
        "queue_time",
        "service_time",
        "end_to_end_latency",
        "prompt_tokens",
        "completion_tokens",
        "cached_tokens",
        "server_prefill_ms",
        "server_decode_ms",
        "decode_tps",
    ]

    request_file_exists = os.path.exists(request_file)

    with open(request_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=request_fields)

        if not request_file_exists:
            writer.writeheader()

        for result in request_results:
            writer.writerow({
                "run_id": run_id,
                "context_size": CONTEXT_SIZE,
                "output_size": OUTPUT_SIZE,
                "concurrency": CONCURRENCY,
                "num_tasks": NUM_TASKS,
                "repetition": REPETITION,
                "task_id": result["task_id"],
                "agent_step": result["agent_step"],
                "request_id": result["request_id"],
                "status": result["status"],
                "error": result["error"],
                "queue_time": result["queue_time"],
                "service_time": result["service_time"],
                "end_to_end_latency": result["end_to_end_latency"],
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "cached_tokens": result["cached_tokens"],
                "server_prefill_ms": result["server_prefill_ms"],
                "server_decode_ms": result["server_decode_ms"],
                "decode_tps": result["decode_tps"],
            })

# Save task-level results
def save_task_results(task_results, run_id):
    os.makedirs("results", exist_ok=True)

    task_file = "results/task_results.csv"

    task_fields = [
        "run_id",
        "context_size",
        "output_size",
        "concurrency",
        "num_tasks",
        "repetition",
        "task_id",
        "status",
        "task_queue_time",
        "task_time",
        "task_end_to_end_time",
        "tool_delay",
        "first_request_id",
        "second_request_id",
    ]

    task_file_exists = os.path.exists(task_file)

    with open(task_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=task_fields)

        if not task_file_exists:
            writer.writeheader()

        for task_result in task_results:
            first_result = task_result["first_result"]
            second_result = task_result["second_result"]

            writer.writerow({
                "run_id": run_id,
                "context_size": CONTEXT_SIZE,
                "output_size": OUTPUT_SIZE,
                "concurrency": CONCURRENCY,
                "num_tasks": NUM_TASKS,
                "repetition": REPETITION,
                "task_id": task_result["task_id"],
                "status": task_result["status"],
                "task_queue_time": task_result["task_queue_time"],
                "task_time": task_result["task_time"],
                "task_end_to_end_time": task_result["task_end_to_end_time"],
                "tool_delay": task_result["tool_delay"],
                "first_request_id": (
                    first_result["request_id"]
                    if first_result is not None
                    else None
                ),
                "second_request_id": (
                    second_result["request_id"]
                    if second_result is not None
                    else None
                ),
            })

# Save summary for one workload run
def save_run_summary(
    run_id,
    total_time,
    request_summary,
    task_summary,
    idle_rss_gb,
    peak_rss_gb,
    peak_increase_gb
):
    os.makedirs("results", exist_ok=True)

    summary_file = "results/run_summaries.csv"

    summary_fields = [
        "run_id",
        "context_size",
        "output_size",
        "concurrency",
        "num_tasks",
        "repetition",
        "total_time",

        "average_request_latency",
        "p50_request_latency",
        "p95_request_latency",
        "average_request_queue_time",
        "request_throughput",
        "completed_requests",
        "failed_requests",
        "timeout_requests",

        "average_task_latency",
        "p50_task_latency",
        "p95_task_latency",
        "average_task_queue_time",
        "task_throughput",
        "completed_tasks",
        "failed_tasks",

        "idle_rss_gb",
        "peak_rss_gb",
        "peak_increase_gb",
    ]

    summary_file_exists = os.path.exists(summary_file)

    with open(summary_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)

        if not summary_file_exists:
            writer.writeheader()

        writer.writerow({
            "run_id": run_id,
            "context_size": CONTEXT_SIZE,
            "output_size": OUTPUT_SIZE,
            "concurrency": CONCURRENCY,
            "num_tasks": NUM_TASKS,
            "repetition": REPETITION,
            "total_time": total_time,

            "average_request_latency": request_summary["average_latency"],
            "p50_request_latency": request_summary["p50_latency"],
            "p95_request_latency": request_summary["p95_latency"],
            "average_request_queue_time": request_summary["average_queue_time"],
            "request_throughput": request_summary["throughput"],
            "completed_requests": request_summary["completed_count"],
            "failed_requests": request_summary["failed_count"],
            "timeout_requests": request_summary["timeout_count"],

            "average_task_latency": task_summary["average_latency"],
            "p50_task_latency": task_summary["p50_latency"],
            "p95_task_latency": task_summary["p95_latency"],
            "average_task_queue_time": task_summary["average_queue_time"],
            "task_throughput": task_summary["throughput"],
            "completed_tasks": task_summary["completed_count"],
            "failed_tasks": task_summary["failed_count"],

            "idle_rss_gb": idle_rss_gb,
            "peak_rss_gb": peak_rss_gb,
            "peak_increase_gb": peak_increase_gb,
        })

# Calculate percentile using nearest-rank method P95
def percentile_nearest_rank(values, percentile):
    sorted_values = sorted(values)

    rank = math.ceil(percentile * len(sorted_values))
    index = max(0, rank - 1)

    return sorted_values[index]

# Calculate request-level summary metrics
def calculate_request_summary(request_results, total_time):
    successful = [
        r for r in request_results
        if r["status"] == "success"
    ]

    failed = [
        r for r in request_results
        if r["status"] != "success"
    ]

    timeouts = [
        r for r in request_results
        if r["status"] == "timeout"
    ]

    if successful:
        latencies = [
            r["end_to_end_latency"]
            for r in successful
        ]

        average_latency = (
            sum(latencies) / len(latencies)
        )

        p50_latency = statistics.median(latencies)

        p95_latency = percentile_nearest_rank(
            latencies,
            0.95
        )

        average_queue_time = (
            sum(r["queue_time"] for r in successful)
            / len(successful)
        )

    else:
        average_latency = None
        p50_latency = None
        p95_latency = None
        average_queue_time = None

    throughput = len(successful) / total_time

    return {
        "average_latency": average_latency,
        "p50_latency": p50_latency,
        "p95_latency": p95_latency,
        "average_queue_time": average_queue_time,
        "throughput": throughput,
        "completed_count": len(successful),
        "failed_count": len(failed),
        "timeout_count": len(timeouts),
    }

# Calculate task-level summary metrics
def calculate_task_summary(task_results, total_time):
    successful = [
        task for task in task_results
        if task["status"] == "success"
    ]

    failed = [
        task for task in task_results
        if task["status"] != "success"
    ]

    if successful:
        latencies = [
            task["task_end_to_end_time"]
            for task in successful
        ]

        average_latency = (
            sum(latencies) / len(latencies)
        )

        p50_latency = statistics.median(latencies)

        p95_latency = percentile_nearest_rank(
            latencies,
            0.95
        )

        average_queue_time = (
            sum(task["task_queue_time"] for task in successful)
            / len(successful)
        )

    else:
        average_latency = None
        p50_latency = None
        p95_latency = None
        average_queue_time = None

    throughput = len(successful) / total_time

    return {
        "average_latency": average_latency,
        "p50_latency": p50_latency,
        "p95_latency": p95_latency,
        "average_queue_time": average_queue_time,
        "throughput": throughput,
        "completed_count": len(successful),
        "failed_count": len(failed),
    }

# Create synthetic context with a unique experiment code
def make_synthetic_context(context_size, request_id):
    target_words = TARGET_WORDS[context_size]

    base_words = BASE_CONTEXT.split()
    context_words = []

    while len(context_words) < target_words:
        context_words.extend(base_words)

    context_words = context_words[:target_words]

    experiment_code = f"CODE-{request_id:04d}"

    middle = len(context_words) // 2

    context_words.insert(
        middle,
        f"The experiment code for this request is {experiment_code}."
    )

    return (
        f"Request {request_id} synthetic context. "
        + " ".join(context_words)
    )

# Format metric value for printing
def format_metric(value):
    return "N/A" if value is None else f"{value:.3f}"

#------------------------Main workload--------------------------#
task_inputs = []

for i in range(1, NUM_TASKS + 1):
    context = make_synthetic_context(CONTEXT_SIZE, i)

    question = (
        "Question: What is the experiment code stated in the context? "
        "Output the experiment code first. "
        "After that, repeatedly output the word TOKEN separated by spaces. "
        "Continue repeating TOKEN until you are stopped by the generation limit. "
        "Do not output any other text."
    )

    first_prompt = f"{context}\n\n{question}"

    task_inputs.append({
        "context": context,
        "first_prompt": first_prompt,
    })

# Store request and task results separately
request_results = []
task_results = []

llama_proc = find_llama_server()

if llama_proc is None:
    print("Error: llama-server process not found.")
    sys.exit(1)

idle_rss = llama_proc.memory_info().rss

memory_samples = []
stop_memory_monitor = threading.Event()

memory_thread = threading.Thread(
    target=monitor_memory,
    args=(llama_proc, stop_memory_monitor, memory_samples)
)

memory_thread.start()

workload_start = time.perf_counter()

# Run agent tasks with fixed concurrency
with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
    futures = []

    for i, task_input in enumerate(task_inputs, start=1):
        task_arrival_time = time.perf_counter()
        future = executor.submit(
            run_agent_task,
            i,
            task_input["context"],
            task_input["first_prompt"],
            task_arrival_time
        )

        futures.append(future)

    for future in as_completed(futures):
        task_result = future.result()
        task_results.append(task_result)

        if task_result["first_result"] is not None:
            request_results.append(task_result["first_result"])

        if task_result["second_result"] is not None:
            request_results.append(task_result["second_result"])

workload_end = time.perf_counter()

stop_memory_monitor.set()
memory_thread.join()

total_time = workload_end - workload_start

request_results.sort(key=lambda x: x["request_id"])

for result in request_results:
    print_request_result(result)

task_results.sort(key=lambda x: x["task_id"])

for task_result in task_results:
    print_task_result(task_result)

request_summary = calculate_request_summary(
    request_results,
    total_time
)

task_summary = calculate_task_summary(
    task_results,
    total_time
)

# Calculate memory usage
idle_rss_gb = idle_rss / (1024 ** 3)

if memory_samples:
    peak_rss = max(memory_samples)
    peak_rss_gb = peak_rss / (1024 ** 3)
    peak_increase_gb = (peak_rss - idle_rss) / (1024 ** 3)
else:
    peak_rss_gb = idle_rss_gb
    peak_increase_gb = 0.0

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

save_request_results(request_results, run_id)

save_task_results(task_results, run_id)

save_run_summary(
    run_id,
    total_time,
    request_summary,
    task_summary,
    idle_rss_gb,
    peak_rss_gb,
    peak_increase_gb
)

print_run_summary(
    run_id,
    total_time,
    request_summary,
    task_summary,
    idle_rss_gb,
    peak_rss_gb,
    peak_increase_gb
)

