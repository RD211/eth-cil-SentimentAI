import os
import gc
import torch
import wandb
import hydra
import torch.nn as nn
from dotenv import load_dotenv
from omegaconf import OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from data_loader import get_dataset, id2label, label2id
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)

load_dotenv()

import torch
from transformers import Trainer

from hydra.core.config_store import ConfigStore
from config.train_classifier import Config as ClassifierConfig
from rag import EmbeddingStore

cs = ConfigStore.instance()
cs.store(name="config", node=ClassifierConfig)


class CustomSentimentTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        final_loss = outputs.get("loss")

        logits = outputs.get("logits")
        probabilities = torch.softmax(logits, dim=-1)
        predicted_sentiment_hard = torch.argmax(probabilities, dim=-1).float()
        mae_metric = torch.mean(torch.abs(labels.float() - predicted_sentiment_hard))
        wandb.log({"score": 1 - 0.5 * mae_metric})

        if return_outputs:
            return final_loss, outputs
        return final_loss


@hydra.main(version_base=None, config_path="config/classifier")
def train(cfg: ClassifierConfig) -> None:
    
    # We merge with default config.
    default_config = OmegaConf.structured(ClassifierConfig)
    cfg = OmegaConf.merge(default_config, cfg)

    # Initialize W&B
    wandb.init(
        project=cfg.logging.project_name,
        name=cfg.run_name,
        config=dict(cfg),
        entity=cfg.logging.entity,
    )

    # Save Hydra config to W&B
    wandb.config.update(dict(cfg))

    # Save train_classifier.py and data_loader.py as artifacts
    wandb.run.log_code(name="train_classifier.py")
    wandb.run.log_code(name="data_loader.py")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    # If no padding token is present, add it
    if tokenizer.pad_token is None:
        print("Adding padding token to tokenizer")
        tokenizer.pad_token = tokenizer.eos_token

    dataset, collator = get_dataset(cfg, tokenizer=tokenizer)



    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.model_name,
        num_labels=3,
        token=os.getenv("HF_TOKEN"),
        id2label=id2label,
        label2id=label2id,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    if cfg.model.use_llm_head_weights:
        print("Loading llm weights for head initialization.")
        llm = AutoModelForCausalLM.from_pretrained(
            cfg.model.llm_head_model_name,
            token=os.getenv("HF_TOKEN"),
            torch_dtype=torch.bfloat16,
        ).to("cuda")

        classes = list(label2id.keys())
        tokens = [tokenizer.encode(c)[-1] for c in classes]

        model.score.weight = nn.Parameter(llm.lm_head.weight[tokens].clone())
        print("Updated weights of model with lm_head.")
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    lora = cfg.model.lora_config

    if lora.enabled:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=lora.r,
            lora_alpha=lora.alpha,
            lora_dropout=lora.dropout,
            target_modules=list(lora.target_modules),
            bias=lora.bias,
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    train_cfg = cfg.train
    logging_cfg = cfg.logging

    trainer = CustomSentimentTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        data_collator=collator,
        args=TrainingArguments(
            max_steps=train_cfg.max_steps,
            run_name=cfg.run_name,
            per_device_train_batch_size=train_cfg.batch_size_per_device,
            per_device_eval_batch_size=train_cfg.batch_size_per_device,
            gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
            num_train_epochs=train_cfg.epochs,
            learning_rate=train_cfg.lr,
            fp16=False,
            bf16=True,
            logging_steps=logging_cfg.logging_steps,
            optim=train_cfg.optimizer,
            lr_scheduler_type=train_cfg.lr_scheduler,
            warmup_steps=train_cfg.warmup_steps,
            seed=cfg.seed,
            max_grad_norm=train_cfg.grad_clip,
            gradient_checkpointing=train_cfg.gradient_checkpointing,
            push_to_hub=True,
            hub_always_push=True,
            hub_model_id=cfg.run_name,
            hub_token=os.getenv("HF_TOKEN"),
            hub_private_repo=True,
            save_strategy='no',
        ),
    )

    trainer.train()
    trainer.push_to_hub()


if __name__ == "__main__":
    train()
