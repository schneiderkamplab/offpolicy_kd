import click
from mltiming import timing
from pathlib import Path

from .offpolicy import distill
from .sampler import ProportionalSampler
from .utils import load_datasets

__all__ = ["main"]

@click.command()
@click.argument('mixture_file', type=click.Path(exists=True))
@click.option('--mixture', default=None, help="Mixture name, if not provided it will be derived from the basename of the mixture file without extension (default: None)")
@click.option('--data-dir', default=None, help="Directory containing the tokenized datasets in Parquet format, if not provided it will be derived from the parent of the mixture file (default: None)")
@click.option('--student', default="models/gemma-3-1b-pt", help="Student model identifier or path (default: models/gemma-3-1b-pt)")
@click.option('--teacher', default="models/gemma-3-4b-pt", help="Teacher model identifier or path (default: models/gemma-3-4b-pt)")
@click.option('--pretrained', is_flag=True, help="Initialize student from pretrained model instead of fresh config (default: False)")
@click.option('--distillation', is_flag=True, help="Do distillation, otherwise it will train without a teacher model (default: False)")
@click.option('--offload-teacher', is_flag=True, help="Offload teacher model to separate GPU during training (default: False)")
@click.option('--seed', default=42, help="Random seed for data shuffling (default: 42)")
@click.option('--alpha', default=1.0, type=float, help="Weight for KL divergence loss in distillation (default: 1.0)")
@click.option('--log-every', default=10, type=int, help="Log training loss every N steps (default: 10)")
@click.option('--collect-every', default=None, type=int, help="Garbage collect every N steps, if not provided it will collect after each validation step (default: None)")
@click.option('--val-every', default=100, type=int, help="Validate every N steps (default: 100)")
@click.option('--val-steps', default=10, type=int, help="Number of validation steps to run (default: 10)")
@click.option('--save-every', default=100, type=int, help="Save model checkpoint every N steps (default: 100)")
@click.option('--save-path', default="checkpoints", help="Directory to save model checkpoints (default: checkpoints)")
@click.option('--save-template', default="student_step{step}.pt", help="Template for saving model checkpoints (default: student_step{step}.pt)")
@click.option('--log-path', default="logs", help="Directory to save training logs (default: logs)")
@click.option('--run-id', default=None, help="Run ID for logging and checkpointing (default: None)")
@click.option('--num-epochs', default=1, type=int, help="Number of training epochs (default: 1)")
@click.option('--patience', default=10, type=int, help="Patience for early stopping (default: 10)")
@click.option('--max-tokens', default=None, type=int, help="Maximum number of training tokens (default: None, meaning no limit)")
@click.option('--max-steps', default=None, type=int, help="Maximum number of training steps (default: None, meaning no limit)")
@click.option('--max-seq-length', default=4096, type=int, help="Maximum sequence length for training (default: 4096)")
@click.option('--gradient-accumulation', default=2, type=int, help="Gradient accumulation steps (default: 2)")
@click.option('--batch-size', default=1, type=int, help="Batch size (default: 1)")
@click.option('--learning-rate', default=1e-5, type=float, help="Learning rate for training (default: 1e-5)")
@click.option('--compile', is_flag=True, help="Compile the model with PyTorch's compile feature (default: False)")
@click.option('--gradient-checkpointing', is_flag=True, help="Enable gradient checkpointing to save memory (default: False)")
def main(**args):
    _main(args, **args)
def _main(args, mixture_file, mixture, data_dir, student, teacher, pretrained, distillation, offload_teacher, seed, alpha, log_every, collect_every, val_every, val_steps, save_every, save_path, save_template, log_path, run_id, num_epochs, patience, max_tokens, max_steps, max_seq_length, gradient_accumulation, batch_size, learning_rate, compile, gradient_checkpointing):
    times = {}
    with timing(times, key="timing/mixture_file"):
        if mixture is None:
            mixture = str(Path(mixture_file).stem)
        if data_dir is None:
            data_dir = Path(mixture_file).parent.parent / "gemma3"
        with open(mixture_file, "rt") as f:
            data_files = [x.strip() for x in f.readline().split(",")]
            weights = [float(x) for x in f.readline().split(",")]
        data_files, weights = zip(*((data_file, weight) for data_file, weight in zip(data_files, weights) if weight))
        train_data_files = [str(data_dir / f"train_{data_file}.parquet") for data_file in data_files]
        val_data_files = [str(data_dir / f"valid_{data_file}.parquet") for data_file in data_files]
    with timing(times, key="timing/load_datasets"):
        train_datasets, val_datasets = load_datasets(train_data_files, val_data_files)
    with timing(times, key="timing/prepare_samplers"):
        train_sampler = ProportionalSampler(train_datasets, weights, seed=seed)
        val_sampler = ProportionalSampler(val_datasets, weights, seed=seed)
    distill(
        args=args,
        times=times,
        experiment=mixture,
        train_datasets=train_datasets,
        val_datasets=val_datasets,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        teacher=teacher,
        student=student,
        pretrained=pretrained,
        distillation=distillation,
        offload_teacher=offload_teacher,
        alpha=alpha,
        log_every=log_every,
        collect_every=collect_every,
        val_every=val_every,
        val_steps=val_steps,
        save_every=save_every,
        save_path=save_path,
        save_template=save_template,
        log_path=log_path,
        run_id=run_id,
        num_epochs=num_epochs,
        patience=patience,
        max_tokens=max_tokens,
        max_steps=max_steps,
        max_seq_length=max_seq_length,
        gradient_accumulation=gradient_accumulation,
        batch_size=batch_size,
        learning_rate=learning_rate,
        compile=compile,
        gradient_checkpointing=gradient_checkpointing,
    )

if __name__ == "__main__":
    main()
