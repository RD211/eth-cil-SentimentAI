import os
import gc
import torch
import wandb
import hydra
import torch.nn as nn
from dotenv import load_dotenv
from omegaconf import DictConfig
from peft import LoraConfig, TaskType, get_peft_model
from data_loader import get_dataset, id2label, label2id
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, TrainingArguments, Trainer
load_dotenv()

import torch
from transformers import Trainer

class CustomSentimentTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        probabilities = torch.softmax(logits, dim=-1)
        
        class_indices = torch.arange(probabilities.size(-1), device=probabilities.device).float()
        predicted_sentiment_soft = torch.sum(probabilities * class_indices, dim=-1)
        
        mae_loss = torch.mean(torch.abs(labels.float() - predicted_sentiment_soft))
        
        base_loss = outputs.get('loss')
        # final_loss = base_loss + 2*mae_loss
        final_loss = base_loss        
        predicted_sentiment_hard = torch.argmax(probabilities, dim=-1).float()
        mae_metric = torch.mean(torch.abs(labels.float() - predicted_sentiment_hard))
        wandb.log({"score": 1 - 0.5 * mae_metric})
        
        if return_outputs:
            return final_loss, outputs
        return final_loss


    
# hydra is a decorator* that reads the config and passes it as an object to the train() function
@hydra.main(version_base=None, config_path="config/classifier", config_name="train")
def train(cfg: DictConfig) -> None:
    
    
    # Initialize W&B
    wandb.login(key=os.getenv('WANDB_API_KEY'))
    wandb.init(project=cfg.logging.project_name, name=cfg.run_name, config=dict(cfg), entity=cfg.logging.entity)
    
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
    
    # this has a randomly initialised classification head, on top of the pretrained base model
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.model_name,
        num_labels=3,
        token=os.getenv('HF_TOKEN'),
        id2label=id2label,
        label2id=label2id,
        torch_dtype=torch.bfloat16,
    ).to('cuda')
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # initialise weight of new llm_head
    if cfg.model.use_llm_head_weights:
        print("Loading llm weights for head initialization.")

        # this has a head that produces logits over the entire vocabulary
        llm = AutoModelForCausalLM.from_pretrained(
            cfg.model.llm_head_model_name,
            token=os.getenv('HF_TOKEN'),
            torch_dtype=torch.bfloat16
        ).to('cuda')
        
        classes = list(label2id.keys())
        # This finds the token ids (from the vocabulary) of the class strings "neutral", "positive" and "negative". If the class has multiple tokens, then it picks the token of the last token. 
        tokens = [tokenizer.encode(c)[-1] for c in classes]

        # then pick the weight values from the LLM with an head over the entire vocabulary, and pick three rows, each corresponding to a token of a class
        model.score.weight = nn.Parameter(llm.lm_head.weight[tokens].clone())
        print("Updated weights of model with lm_head.")

        # remove the model as we no longer need it
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        
    
    lora = cfg.model.lora_config

    # Use LoRa
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
        # eval_dataset=dataset["test"],
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
            # weight_decay=train_cfg.weight_decay,
            lr_scheduler_type=train_cfg.lr_scheduler,
            warmup_steps=train_cfg.warmup_steps,
            seed=cfg.seed,
            # output_dir=train_cfg.output_folder,
            max_grad_norm=train_cfg.grad_clip,
            gradient_checkpointing=train_cfg.gradient_checkpointing,
            push_to_hub=True,
            hub_always_push=True,
            hub_model_id=cfg.run_name,
            hub_token=os.getenv('HF_TOKEN'),
            hub_private_repo=True,
        ),
    )
    
    trainer.train()
    trainer.push_to_hub()
    # Evaluate the model
    # eval_result = trainer.evaluate()
    # print(eval_result)
    
if __name__ == "__main__":
    train()