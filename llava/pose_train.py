import os
import logging
import argparse
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    HfArgumentParser,
    TrainingArguments
)
from datasets import load_from_disk
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.train.llava_trainer import LLaVATrainer

from llava.pose_encoder import PoseEncoder
from llava.pose_data_collator import PoseDataCollator
from llava.pose_llava_model import PoseLlavaModel
import pathlib

# Set up logging
logger = logging.getLogger(__name__)

class PoseDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, dataset=None, data_path=None, data_args=None):
        if dataset is not None:
            self.data = dataset
        elif data_path is not None:
            self.data = load_from_disk(data_path)["train"]
        else:
            raise ValueError("Either dataset or data_path must be provided.")
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        print('len(self.train_dataset)', len(self.data))
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="lmsys/vicuna-7b-v1.5")
    version: Optional[str] = field(default="v1")
    freeze_backbone: bool = field(default=False)
    tune_mm_projector_only: bool = field(default=True)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the pose data"})
    lazy_preprocess: bool = field(default=False)

@dataclass
class PoseTrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = field(default=1e-4)
    group_by_modality_length: bool = field(default=False)


def train():
    print('Training...')
    parser = HfArgumentParser((ModelArguments, DataArguments, PoseTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print('Data args:', data_args)
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler()],
    )
    
    # Set the verbosity of the logger based on the training args
    if training_args.should_log:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Load base LLM
    model_name = get_model_name_from_path(model_args.model_name_or_path)
    tokenizer, base_model, _, context_len = load_pretrained_model(
        model_args.model_name_or_path, 
        None, 
        model_name
    )
    
    # Create pose encoder
    pose_encoder = PoseEncoder(base_model.config)
    
    # Create PoseLlavaModel
    model = PoseLlavaModel(
        base_model,
        pose_encoder,
        projector_type=model_args.mm_projector_type
    )
    
    # Freeze LLM backbone if needed
    if model_args.freeze_backbone:
        model.base_model.requires_grad_(False)
    
    # Freeze all parameters except for the projector if tune_mm_projector_only is True
    if model_args.tune_mm_projector_only:
        model.requires_grad_(False)
        model.mm_projector.requires_grad_(True)
    
    # Load pretrained projector weights if provided
    if model_args.pretrain_mm_mlp_adapter is not None:
        mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
        missing, unexpected = model.mm_projector.load_state_dict(mm_projector_weights, strict=False)
        
        if len(missing) > 0:
            logger.warning(f"Missing keys in projector weights: {missing}")
        if len(unexpected) > 0:
            logger.warning(f"Unexpected keys in projector weights: {unexpected}")
    
    # Load HF Arrow dataset with train/validation splits
    dataset = load_from_disk(data_args.data_path)

    # Optionally wrap with PoseDataset logic if needed
    train_dataset = PoseDataset(tokenizer=tokenizer, dataset=dataset["train"], data_args=data_args)
    eval_dataset = PoseDataset(tokenizer=tokenizer, dataset=dataset["validation"], data_args=data_args)

    
    data_collator = PoseDataCollator(tokenizer=tokenizer)
    
    # Create trainer
    trainer = LLaVATrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Training
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    # Save the trained model
    trainer.save_state()
    
    # Save the mm_projector separately
    if model_args.tune_mm_projector_only:
        os.makedirs(training_args.output_dir, exist_ok=True)
        torch.save(
            model.mm_projector.state_dict(),
            os.path.join(training_args.output_dir, 'mm_projector.bin')
        )
    
    # Save the entire model
    model.config.save_pretrained(training_args.output_dir)
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    train()