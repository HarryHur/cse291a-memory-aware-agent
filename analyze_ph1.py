import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_PATH = "results/run_summaries.csv"
PLOTS_DIR = "plots"


def load_data():
    return pd.read_csv(RESULTS_PATH)


def run_qa(df):
    print("=== Phase 1 Data QA ===")
    print(f"Total runs: {len(df)}")

    print("\nRuns by repetition:")
    print(df["repetition"].value_counts().sort_index())

    print("\nRuns by condition:")
    print(
        df.groupby(
            ["context_size", "output_size", "concurrency"]
        ).size().sort_index()
    )


def build_summary(df):
    metrics = [
        "total_time",
        "average_request_latency",
        "p95_request_latency",
        "average_task_latency",
        "p95_task_latency",
        "average_task_queue_time",
        "task_throughput",
        "peak_rss_gb",
        "peak_increase_gb",
    ]

    return (
        df.groupby(
            ["context_size", "output_size", "concurrency"]
        )[metrics]
        .agg(["mean", "std"])
        .round(3)
    )


def print_readable_summary(df):
    metrics = [
        "total_time",
        "average_request_latency",
        "average_task_latency",
        "average_task_queue_time",
        "task_throughput",
        "peak_rss_gb",
    ]

    readable = (
        df.groupby(
            ["context_size", "output_size", "concurrency"]
        )[metrics]
        .mean()
        .round(3)
        .reset_index()
    )

    print("\n=== Readable Phase 1 Summary ===")
    print(readable.to_string(index=False))

def plot_throughput_vs_concurrency(df):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Controlled comparison: keep output size fixed at "short".
    data = df[df["output_size"] == "short"]

    context_order = ["short", "medium", "long"]
    concurrency_order = [1, 2, 4]

    plt.figure(figsize=(7, 5))

    for context in context_order:
        subset = data[data["context_size"] == context]

        stats = (
            subset.groupby("concurrency")["task_throughput"]
            .agg(["mean", "std"])
            .reindex(concurrency_order)
        )

        plt.errorbar(
            stats.index,
            stats["mean"],
            yerr=stats["std"],
            marker="o",
            capsize=4,
            label=context,
        )

    plt.xlabel("Concurrency")
    plt.ylabel("Task Throughput (tasks/s)")
    plt.title("Task Throughput vs Concurrency (Short Output)")
    plt.xticks(concurrency_order)
    plt.legend(title="Context Size")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = f"{PLOTS_DIR}/throughput_vs_concurrency.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"\nSaved: {output_path}")

def plot_request_latency_vs_concurrency(df):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Controlled comparison: keep output size fixed at "short".
    data = df[df["output_size"] == "short"]

    context_order = ["short", "medium", "long"]
    concurrency_order = [1, 2, 4]

    plt.figure(figsize=(7, 5))

    for context in context_order:
        subset = data[data["context_size"] == context]

        stats = (
            subset.groupby("concurrency")["average_request_latency"]
            .agg(["mean", "std"])
            .reindex(concurrency_order)
        )

        plt.errorbar(
            stats.index,
            stats["mean"],
            yerr=stats["std"],
            marker="o",
            capsize=4,
            label=context,
        )

    plt.xlabel("Concurrency")
    plt.ylabel("Average Request Latency (s)")
    plt.title("Request Latency vs Concurrency (Short Output)")
    plt.xticks(concurrency_order)
    plt.legend(title="Context Size")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = f"{PLOTS_DIR}/request_latency_vs_concurrency.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")

def plot_task_queue_vs_concurrency(df):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Controlled comparison: keep output size fixed at "short".
    data = df[df["output_size"] == "short"]

    context_order = ["short", "medium", "long"]
    concurrency_order = [1, 2, 4]

    plt.figure(figsize=(7, 5))

    for context in context_order:
        subset = data[data["context_size"] == context]

        stats = (
            subset.groupby("concurrency")["average_task_queue_time"]
            .agg(["mean", "std"])
            .reindex(concurrency_order)
        )

        plt.errorbar(
            stats.index,
            stats["mean"],
            yerr=stats["std"],
            marker="o",
            capsize=4,
            label=context,
        )

    plt.xlabel("Concurrency")
    plt.ylabel("Average Task Queue Time (s)")
    plt.title("Task Queue Time vs Concurrency (Short Output)")
    plt.xticks(concurrency_order)
    plt.legend(title="Context Size")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = f"{PLOTS_DIR}/task_queue_vs_concurrency.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")

def plot_memory_vs_context(df):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Controlled comparison:
    # keep concurrency=1 and output_size="short".
    data = df[
        (df["output_size"] == "short")
        & (df["concurrency"] == 1)
    ]

    context_order = ["short", "medium", "long"]

    stats = (
        data.groupby("context_size")["peak_rss_gb"]
        .agg(["mean", "std"])
        .reindex(context_order)
    )

    plt.figure(figsize=(7, 5))

    plt.errorbar(
        context_order,
        stats["mean"],
        yerr=stats["std"],
        marker="o",
        capsize=4,
    )

    plt.xlabel("Context Size")
    plt.ylabel("Peak RSS (GiB)")
    plt.title("Peak Memory vs Context Size (Concurrency = 1, Short Output)")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = f"{PLOTS_DIR}/memory_vs_context.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")

def plot_throughput_by_output_size_vs_concurrency(df):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Controlled comparison:
    # keep context size fixed at "medium".
    data = df[df["context_size"] == "medium"]

    output_order = ["short", "long"]
    concurrency_order = [1, 2, 4]

    plt.figure(figsize=(7, 5))

    for output_size in output_order:
        subset = data[data["output_size"] == output_size]

        stats = (
            subset.groupby("concurrency")["task_throughput"]
            .agg(["mean", "std"])
            .reindex(concurrency_order)
        )

        plt.errorbar(
            stats.index,
            stats["mean"],
            yerr=stats["std"],
            marker="o",
            capsize=4,
            label=output_size,
        )

    plt.xlabel("Concurrency")
    plt.ylabel("Task Throughput (tasks/s)")
    plt.title("Task Throughput vs Concurrency (Medium Context)")
    plt.xticks(concurrency_order)
    plt.legend(title="Output Size")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = f"{PLOTS_DIR}/throughput_by_output_size_vs_concurrency.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")

def plot_throughput_vs_output_size(df):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Controlled comparison:
    # keep context size fixed at "medium".
    data = df[df["context_size"] == "medium"]

    output_order = ["short", "long"]
    concurrency_order = [1, 2, 4]

    plt.figure(figsize=(7, 5))

    for concurrency in concurrency_order:
        subset = data[data["concurrency"] == concurrency]

        stats = (
            subset.groupby("output_size")["task_throughput"]
            .agg(["mean", "std"])
            .reindex(output_order)
        )

        plt.errorbar(
            output_order,
            stats["mean"],
            yerr=stats["std"],
            marker="o",
            capsize=4,
            label=f"C={concurrency}",
        )

    plt.xlabel("Output Size")
    plt.ylabel("Task Throughput (tasks/s)")
    plt.title("Task Throughput vs Output Size (Medium Context)")
    plt.legend(title="Concurrency")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = f"{PLOTS_DIR}/throughput_vs_output_size.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")

def plot_request_latency_vs_output_size(df):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Controlled comparison:
    # keep context size fixed at "medium".
    data = df[df["context_size"] == "medium"]

    output_order = ["short", "long"]
    concurrency_order = [1, 2, 4]

    plt.figure(figsize=(7, 5))

    for concurrency in concurrency_order:
        subset = data[data["concurrency"] == concurrency]

        stats = (
            subset.groupby("output_size")["average_request_latency"]
            .agg(["mean", "std"])
            .reindex(output_order)
        )

        plt.errorbar(
            output_order,
            stats["mean"],
            yerr=stats["std"],
            marker="o",
            capsize=4,
            label=f"C={concurrency}",
        )

    plt.xlabel("Output Size")
    plt.ylabel("Average Request Latency (s)")
    plt.title("Request Latency vs Output Size (Medium Context)")
    plt.legend(title="Concurrency")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = f"{PLOTS_DIR}/request_latency_vs_output_size.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")

def main():
    df = load_data()

    run_qa(df)

    summary = build_summary(df)

    print("\n=== Mean / Std across 3 repetitions ===")
    print(summary.to_string())

    print_readable_summary(df)

    plot_throughput_vs_concurrency(df)
    plot_request_latency_vs_concurrency(df)
    plot_task_queue_vs_concurrency(df)
    plot_memory_vs_context(df)
    plot_throughput_by_output_size_vs_concurrency(df) 
    plot_throughput_vs_output_size(df)
    plot_request_latency_vs_output_size(df)

if __name__ == "__main__":
    main()