from dataclasses import dataclass, field
from typing import List

@dataclass
class LoraConfig:
    enabled: bool = False
    r: int = 32
    alpha: int = 32
    bias: str = "none"
    dropout: float = 0.01
    target_modules: List[str] = field(
        default_factory=lambda: [
            "v_proj",
            "o_proj",
            "k_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass
class ModelConfig:
    model_name: str = "HuggingFaceTB/SmolLM2-1.7B"
    use_llm_head_weights: bool = False
    llm_head_model_name: str = "HuggingFaceTB/SmolLM2-1.7B"
    lora_config: LoraConfig = field(default_factory=LoraConfig)

    is_instruct: bool = False


@dataclass
class TrainConfig:
    lr: float = 5e-5
    epochs: int = 1
    max_steps: int = -1
    batch_size_per_device: int = 64
    gradient_accumulation_steps: int = 1
    optimizer: str = "paged_adamw_32bit"
    weight_decay: float = 0.0
    lr_scheduler: str = "cosine"
    warmup_steps: int = 128
    grad_clip: float = 1.0
    gradient_checkpointing: bool = True


@dataclass
class DataConfig:
    path: str = "./data/train"
    test_size: float = 0.0
    prompt: str = "prompt_templates/sentiment.txt"
    seed: int = 42
    num_proc: int = 8
    max_seq_length: int = 1024


@dataclass
class LoggingConfig:
    wandb: bool = True
    logging_steps: int = 1
    project_name: str = "Sentiments"
    entity: str = "cil-sentiment-ai"

@dataclass
class RAGConfig:
    use_rag: bool = False
    k: int = 5
    out_of: int = 5
    embedding_model: str = "jinaai/jina-embeddings-v3"
    data_percentage_reserve: float = 0.25

@dataclass
class Config:
    run_name: str = "SmolLM2-1.7b-Instruct"
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    seed: int = 42
    defaults: List[str] = field(default_factory=lambda: ["_self_"])
