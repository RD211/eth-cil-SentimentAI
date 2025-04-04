from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class LoraConfig:
    enable: bool = False
    rank: int = 256
    alpha: float = 512
    target_modules: Any = "all-linear"
    dropout: float = 0.01
    bias: str = 'none'

@dataclass
class ModelConfig:
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_length: int = 500
    lora: LoraConfig = field(default_factory=LoraConfig)

@dataclass
class GenerationConfig:
    prompt_template_path: str = "prompt_templates/sentiment.txt"
    use_code_eval: bool = False
    temperature: float = 0.7
    top_k: int = 2**16
    top_p: float = 1.0
    
@dataclass
class Dataset:
    name_or_path: str = "./data/train"
    split: str = "train"
    ratio: float = 1.0

@dataclass
class DatasetConfig:
    dataset: list[Dataset] = field(default_factory=list)
    max_examples: Optional[int] = None
    skip: int = 0

@dataclass
class HuggingFaceConfig:
    name: str = "eth-text-classification/qwen2.5-0.5b-instruct"
    push_to_hub: bool = False

@dataclass
class LoggingConfig:
    wandb: bool = False
    wandb_project: str = "train-rl"
    wandb_run_name: str = "Qwen2.5-0.5B-Instruct"
    wandb_entity: str = "eth-text-classification"
    run_group: str = "0.5b"
    wandb_tags: list[str] = field(default_factory=list)
    save_dir: str = "output"
    save_steps: int = 10

@dataclass
class TrainConfig:
    gradient_checkpointing: bool = True
    num_samples_per_problem: int = 8
    number_of_problems_per_batch: int = 4

    lr_scheduler_type: str = "constant"
    optimizer: str = "adamw_hf"
    epochs: int = 1
    max_steps: int = -1
    deepspeed_config_path: Optional[str] = None

    beta: float = 0.001
    learning_rate: float = 5e-7
    mu: int = 1
    epsilon: float = 0.2
    sync_ref_model: bool = False
    ref_model_mixup_alpha: float = 0.9
    ref_model_sync_steps: int = 64


@dataclass
class RLModelTrainingConfig:
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    seed: int = 42