import os
import json
import click
import matplotlib.pyplot as plt
from math import pow

def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]

def extract_metric(data, metric, scale):
    steps = []
    values = []
    for d in data:
        if metric not in d or "step" not in d:
            continue
        steps.append(d["step"])
        values.append(d[metric]*scale)
    return steps, values

@click.command()
@click.argument('run_dirs', nargs=-1, type=click.Path(exists=True, file_okay=False))
@click.option('--metrics', default="loss", help='Comma-separated list of metrics to plot (e.g., "loss,ce_loss,kl_loss,ppl") [default: loss]')
@click.option('--scale', default=1.0, help='Scale factor for chosen metrics [default: 1.0]')
def plot_logs(run_dirs, metrics, scale):
    """
    For each RUN_DIR, load train.jsonl, val.jsonl, and stdout.txt to:
    - Extract metrics (losses, ppl, etc.)
    - Save the plot to <run_dir>_<metrics>.png
    """
    metric_list = [m.strip() for m in metrics.split(",") if m.strip()]
    metric_suffix = ",".join(metric_list)
    for run_path in run_dirs:
        train_path = os.path.join(run_path, "train.jsonl")
        val_path = os.path.join(run_path, "val.jsonl")
        stdout_path = os.path.join(run_path, "stdout.txt")
        if not os.path.isfile(train_path) or not os.path.isfile(val_path):
            click.echo(f"[WARN] Skipping {run_path}: train/val logs not found.")
            continue
        try:
            train_data = read_jsonl(train_path)
            val_data = read_jsonl(val_path)
            plt.figure()
            found_any = False
            for metric in metric_list:
                t_steps, t_vals = extract_metric(train_data, metric, scale)
                v_steps, v_vals = extract_metric(val_data, metric, scale)
                if t_steps and t_vals:
                    plt.plot(t_steps, t_vals, label=f"Train {metric}")
                    found_any = True
                else:
                    click.echo(f"[WARN] {run_path}: No train data for metric '{metric}'")
                if v_steps and v_vals:
                    plt.plot(v_steps, v_vals, label=f"Val {metric}")
                    found_any = True
                else:
                    click.echo(f"[WARN] {run_path}: No val data for metric '{metric}'")
            if not found_any:
                click.echo(f"[WARN] No valid data in {run_path}. Skipping plot.")
                plt.close()
                continue
            plt.xlabel("Step")
            plt.ylabel("Scaled Metric Value")
            plt.title(f"{os.path.basename(run_path)}: {metric_suffix} (scaled)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            out_path = f"{run_path}_{metric_suffix}.png"
            plt.savefig(out_path)
            click.echo(f"[INFO] Saved plot: {out_path}")
            plt.close()
        except Exception as e:
            click.echo(f"[ERROR] Failed to process {run_path}: {e}")

if __name__ == '__main__':
    plot_logs()
