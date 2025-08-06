import click
from mltiming import timing

from .offpolicy import distill
from .sampler import RandomSampler
from .utils import load_datasets
import numpy as np
import json

__all__ = ["main"]

def validate_distribution(ctx, param, value):
    try:
        # Convert JSON string to Python list of lists
        dist = json.loads(value)
        # Convert to NumPy array
        dist_array = np.array(dist, dtype=np.float64)
    except Exception as e:
        raise click.BadParameter(f"Invalid input. Must be a JSON list of lists of floats. Error: {e}")

    # Check shape
    if dist_array.ndim != 2 or dist_array.shape[1] != 4:
        raise click.BadParameter(f"Each row must have exactly 4 elements. Got shape: {dist_array.shape}")
    

    # Check row-wise sum
    row_sums = dist_array.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        bad_rows = np.where(~np.isclose(row_sums, 1.0, atol=1e-6))[0]
        raise click.BadParameter(f"Rows {bad_rows.tolist()} must sum to 1.0. Row sums: {row_sums.tolist()}")

    return dist_array.tolist()  # Now validated and ready as a NumPy array


@click.command()
@click.argument('train_data_files', nargs=-1, type=click.Path(exists=True))
@click.option('--val-data-files', multiple=True, type=click.Path(exists=True), default=None, help="Validation data files in Parquet format, if not provided it will use the same files as training data")
@click.option('--experiment', default=None, help="Experiment name (default: None)")
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
@click.option('--warmup-steps', default=None, type=float, help="warm-upsteps in percentage  (default: None)")
@click.option('--max-seq-length', default=4096, type=int, help="Maximum sequence length for training (default: 4096)")
@click.option('--gradient-accumulation', default=2, type=int, help="Gradient accumulation steps (default: 2)")
@click.option('--batch-size', default=1, type=int, help="Batch size (default: 1)")
@click.option('--learning-rate', default=1e-5, type=float, help="Learning rate for training (default: 1e-5)")
@click.option('--compile', is_flag=True, help="Compile the model with PyTorch's compile feature (default: False)")
@click.option('--gradient-checkpointing', is_flag=True, help="Enable gradient checkpointing to save memory (default: False)")
@click.option('--offload-optimizer', is_flag=True, help="Offload optimizer state to CPU to save GPU memory (default: False)")
@click.option('--overwrite', is_flag=True, help="Overwrite existing checkpoints and logs (default: False)")
@click.option('--yes', is_flag=True, help="Automatically answer yes to prompts (default: False)")
@click.option('--attn-implementation', default='eager', type=click.Choice(['eager', 'flash_attention', 'flash_attention_2', 'mem_efficient'], case_sensitive=False), help="Attention implementation to use (default: eager)")
@click.option('--lr-scheduler-type', default="cosine", type=click.Choice(['linear', 'cosine', 'cosine_with_restarts']))
@click.option('--evaluate-only', is_flag=True, help="Only evaluate the model without training (default: False)")
@click.option('--load-checkpoint', type=click.Path(exists=True), help="Path to a checkpoint to load the model from (default: None)")
@click.option('--collate-type', default="truncate", type=click.Choice(['truncate', 'pack']), help="Collate function type to use for batching (default: truncate)")
@click.option('--on_policy', is_flag=True, help="Do *on policy* distillation, use only with distillation flag (default: False)")
@click.option('--distribution',callback=validate_distribution, help='JSON string for list of lists (e.g., [[0.25,0.25,0.25,0.25],[0.5,0.5,0,0]])'
)

def main(**args):
    _main(args, **args)
def _main(args, train_data_files, val_data_files, experiment, student, teacher, pretrained, distillation, offload_teacher, seed, alpha, log_every, collect_every, val_every, val_steps, save_every, save_path, save_template, log_path, run_id, num_epochs, patience, max_tokens, max_steps, warmup_steps, max_seq_length, gradient_accumulation, batch_size, learning_rate, compile, gradient_checkpointing, offload_optimizer, overwrite, yes, attn_implementation, lr_scheduler_type, evaluate_only, load_checkpoint, collate_type, on_policy, distribution):
    
    distribution = np.array(distribution)

    # Validate that distribution rows match num_epochs or is a single row
    if distribution.shape[0] not in (1, num_epochs):
        raise click.ClickException(
            f"Number of rows in --distribution ({distribution.shape[0]}) must be 1 or equal to --num-epochs ({num_epochs})."
        )
    
    if not isinstance(distribution, tuple) or len(distribution) != 4:
        raise click.BadParameter("Must be a tuple of four floats.")
    
    
    total = sum(distribution)
    if abs(total - 1.0) > 1e-6:  # allowing for floating point tolerance
        raise click.BadParameter("The distribution values must sum to 1.0.")
    
    times = {}
    with timing(times, key="timing/load_datasets"):
        print("Loading datasets...")
        train_datasets, val_datasets = load_datasets(train_data_files, val_data_files, evaluate_only)
    with timing(times, key="timing/prepare_samplers"):
        train_sampler = None if evaluate_only else RandomSampler(train_datasets, seed=seed)
        val_sampler = RandomSampler(val_datasets, seed=seed)
    distill(
        args=args,
        times=times,
        experiment=experiment,
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
        warmup_steps=warmup_steps,
        max_seq_length=max_seq_length,
        gradient_accumulation=gradient_accumulation,
        batch_size=batch_size,
        learning_rate=learning_rate,
        compile=compile,
        gradient_checkpointing=gradient_checkpointing,
        offload_optimizer=offload_optimizer,
        overwrite=overwrite,
        yes=yes,
        attn_implementation=attn_implementation,
        lr_scheduler_type=lr_scheduler_type,
        evaluate_only=evaluate_only,
        load_checkpoint=load_checkpoint,
        collate_type=collate_type,
        on_policy=on_policy,
        distribution=distribution,
    )

if __name__ == "__main__":
    main()
