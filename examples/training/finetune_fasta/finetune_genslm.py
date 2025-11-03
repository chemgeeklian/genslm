from __future__ import annotations

from Bio import SeqIO
import os
from argparse import ArgumentParser
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
#from accelerate import Accelerator

import torch
import transformers
import wandb
import yaml
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

PathLike = str | Path
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

@dataclass
class Sequence:
    """Store a biological sequence and its description tag."""

    sequence: str
    tag: str

    def __hash__(self) -> int:
        """Hash the sequence and tag."""
        return hash((self.sequence, self.tag))

def read_fasta(path):
    return [Sequence(str(record.seq), record.description) for record in SeqIO.parse(path, "fasta")]


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """TrainingArguments for configuring the Hugging Face Trainer.

    Here we provide some sensible defaults for the arguments for our use case.
    """

    output_dir: str = field(
        default='test_run',
        metadata={
            'help': 'The output directory where the model predictions and '
            'checkpoints will be written.'
        },
    )
    per_device_train_batch_size: int = field(
        default=64,
        metadata={'help': 'Batch size per GPU/TPU core/CPU for training.'},
    )
    per_device_eval_batch_size: int = field(
        default=128,
        metadata={'help': 'Batch size per GPU/TPU core/CPU for evaluation.'},
    )
    num_train_epochs: float = field(
        default=20,
        metadata={'help': 'Total number of training epochs to perform.'},
    )
    learning_rate: float = field(
        default=4e-4,
        metadata={'help': 'The initial learning rate for Adam.'},
    )
    warmup_steps: int = field(
        default=1_000,
        metadata={'help': 'Linear warmup over `warmup_steps`.'},
    )
    lr_scheduler_type: str = field(
        default='cosine',
        metadata={'help': 'The scheduler type to use.'},
    )
    weight_decay: float = field(
        default=0.01,
        metadata={'help': 'The weight decay to apply.'},
    )
    eval_steps: int = field(
        default=500,
        metadata={
            'help': 'Number of steps between evaluations. If `eval_steps` '
            'is modified, update `logging_steps` and `save_steps` to the same '
            'value.'
        },
    )
    save_total_limit: int = field(
        default=1,
        metadata={'help': 'Total number of checkpoints to save.'},
    )
    save_strategy: str = field(
        default='steps',
        metadata={'help': 'Strategy for saving checkpoints.'},
    )
    evaluation_strategy: str = field(
        default='steps',
        metadata={'help': 'Strategy for evaluating.'},
    )
    load_best_model_at_end: bool = field(
        default=True,
        metadata={
            'help': 'Whether to load the best model at the end of training. '
            'When `save_total_limit` is set to 1, will save the best model as '
            'well as the last model if the last model is worse (eval_loss) '
            'than the best model.'
        },
    )
    fp16: bool = field(
        default=True,
        metadata={'help': 'Whether to use 16-bit (mixed) precision training.'},
    )
    dataloader_num_workers: int = field(
        default=4,
        metadata={'help': 'Number of subprocesses to use for data loading.'},
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={
            'help': 'This skips underlying logic in Trainer which modifies '
            'the data_collator (do not change).'
        },
    )


@dataclass
class TrainingConfig:
    """Configuration for fine tuning the ESM model."""

    train_path: str = field(
        metadata={'help': 'Path to training data.'},
    )
    eval_path: str = field(
        metadata={'help': 'Path to validation data.'},
    )
    training_args: TrainingArguments = field(
        default_factory=TrainingArguments,
        metadata={
            'help': 'Hugging face arguments for training the model '
            '(see transformers.TrainingArguments).'
        },
    )
    wandb_project: str = field(
        default='',
        metadata={
            'help': 'Wandb project name (By default, set to empty string'
            ' to turn off wandb).'
        },
    )
    model_name: str = field(
        default='genslm_25M_patric',
        metadata={'help': 'Name of the GenSLM model to use.'},
    )
    model_cache_dir: str = field(
        default='',
        metadata={'help': 'Directory to cache the model.'},
    )

    def __post_init__(self) -> None:
        """Initialize the training arguments and log the config."""
        # Populate the training arguments
        self.training_args = TrainingArguments(**self.training_args)

        # Set the output directory
        output_dir = Path(self.training_args.output_dir)

        # Create the output directory if it doesn't exist
        output_dir.mkdir(exist_ok=True, parents=True)

        # wandb needs to be initialized once on all node ranks
        if self.wandb_project and self.training_args.local_process_index == 0:
            os.environ['WANDB_PROJECT'] = self.wandb_project
            # Assign the same group name as the output directory
            # so that multi-node runs are grouped together
            wandb.init(dir=output_dir, group=output_dir.name)
            wandb.config.update(
                {'train_config': asdict(self)}, allow_val_change=True
            )

        # Set the report_to argument based on the wandb project
        self.training_args.report_to = ['wandb' if self.wandb_project else '']

        # Log the config to a yaml file
        with open(output_dir / 'train_config.yaml', 'w') as fp:
            yaml.dump(asdict(self), fp)


class ClearEvalMemoryTrainer(Trainer): 
    """Trainer that clears the cuda cache before each evaluation.

    Note: reduces OOMs for some models.
    """

    def _clear_cuda_cache(self) -> None:
        import gc

        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()
            torch.clear_autocast_cache()

    def evaluate(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        """Clear the cuda cache before evaluation."""
        self._clear_cuda_cache()
        return super().evaluate(*args, **kwargs)


def main() -> None:
    from genslm import GenSLM, SequenceDataset

    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    # Load the training configuration
    with open(args.config) as fp:
        config = TrainingConfig(**yaml.safe_load(fp))

    model = GenSLM(config.model_name, model_cache_dir=config.model_cache_dir)
    #model = GenSLM("genslm_2.5B_patric", model_cache_dir="/lus/flare/projects/FoundEpidem/xlian/models/genslm_models/2.5B")
    # model.load_state_dict(torch.load(
    #     '/lus/flare/projects/FoundEpidem/xlian/verAB_genslm/genslm/runs_prod/nolog-4gpu-1ep-van/checkpoint-2929/pytorch_model_fsdp.bin',
    #     map_location='cpu',
    #     ), strict=False)

    tokenizer = model.tokenizer 
    tokenizer.model_max_length = 1024 #model.config.max_position_embeddings

    # Read the fasta file into a list of sequences
    train_sequences = [seq.sequence for seq in read_fasta(config.train_path)]#[:10000]
    eval_sequences = [seq.sequence for seq in read_fasta(config.eval_path)]#[:100]

    train_dataset = SequenceDataset(train_sequences, tokenizer, seq_length=tokenizer.model_max_length, kmer_size=3)
    eval_dataset = SequenceDataset(eval_sequences, tokenizer, seq_length=tokenizer.model_max_length, kmer_size=3)

    trainer = ClearEvalMemoryTrainer(
        model=model,
        args=config.training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Attempt to load a checkpoint
    '''
    checkpoint = get_last_checkpoint(config.training_args.output_dir)
    if checkpoint is not None:
        print('Training from checkpoint:', checkpoint)

    # Train the model
    
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    '''

    train_result = trainer.train()
    trainer.save_model()
    metrics = train_result.metrics
    trainer.log_metrics('train', metrics)
    trainer.save_metrics('train', metrics)
    trainer.save_state()
    
if __name__ == '__main__':
    import multiprocessing as mp
    #mp.set_start_method("spawn", force=True)

    main()

# python finetune_genslm.py --config config.yaml 