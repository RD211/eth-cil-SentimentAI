import hydra
import wandb
import torch
from datetime import timedelta
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
from config.train_rl_model import RLModelTrainingConfig 
from transformers import set_seed
from dotenv import load_dotenv
from accelerate import Accelerator, InitProcessGroupKwargs
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from datasets import load_dataset, concatenate_datasets, Dataset, load_from_disk
from trl import GRPOConfig
from custom_grpo import CustomGRPOTrainer
import warnings
warnings.filterwarnings("ignore")
load_dotenv()
import os
os.environ['VLLM_USE_V1']='0'

# logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="config", node=RLModelTrainingConfig)

@hydra.main(config_path="config/train_rl", version_base=None)
def main(cfg: RLModelTrainingConfig):

    kwargs = [InitProcessGroupKwargs(timeout=timedelta(hours=10))]
    accelerator = Accelerator(kwargs_handlers=kwargs)

    # Configs
    model_config = cfg.model
    train_config = cfg.train
    logging_config = cfg.logging
    lora_config = model_config.lora
    data_config = cfg.dataset

    set_seed(cfg.seed)

    if logging_config.wandb and accelerator.is_local_main_process:
        wandb.init(
            project=logging_config.wandb_project,
            name=logging_config.wandb_run_name,
            entity=logging_config.wandb_entity,
            group=logging_config.run_group,
            tags=logging_config.wandb_tags,
            config=OmegaConf.to_object(cfg),
        )
        include_fn = lambda x: 'output' not in x and (x.endswith('.py') or x.endswith('.yaml') or x.endswith('.txt'))
        wandb.run.log_code('.', include_fn=include_fn)

    logger.info('Loading model')
    if accelerator.is_main_process:
        try:
            snapshot_download(model_config.model_name_or_path)
        except:
            print("Could not do snapshot download. No worries though.")
    accelerator.wait_for_everyone()

    torch_dtype = torch.bfloat16
    model_kwargs = dict(
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch_dtype,
        use_cache=False if train_config.gradient_checkpointing else True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, trust_remote_code=True)

    peft_config = None
    if lora_config.enable:
        peft_config = LoraConfig(
          task_type=TaskType.CAUSAL_LM,
          r = lora_config.rank,
          lora_alpha=lora_config.alpha,
          target_modules=lora_config.target_modules,
          lora_dropout=lora_config.dropout,
          bias=lora_config.bias
        )

    if accelerator.is_main_process:
        logger.info('Getting dataset')

    # Load the datasets
    datasets = []
    for dataset in cfg.dataset.datasets:
        try:
          dataset = load_dataset(dataset.name_or_path, split=dataset.split)
        except:
            dataset = load_from_disk(dataset.name_or_path)
        datasets.append(dataset)

    # We sample based on max_examples and ratios.
    if cfg.dataset.max_examples is not None:
        ratios = [dataset.ratio for dataset in cfg.dataset.datasets]
        total_ratio = sum(ratios)
        num_samples = cfg.dataset.max_examples

        # Sample based on ratios
        if num_samples == -1:
            for i, dataset in enumerate(datasets):
              datasets[i] = dataset.shuffle(
                  seed=cfg.seed
              )
        else:
          samples_per_dataset = [int(num_samples * ratio / total_ratio) for ratio in ratios]
          for i, dataset in enumerate(datasets):
              datasets[i] = dataset.shuffle(
                  seed=cfg.seed
              ).select(range(samples_per_dataset[i]))

    # Concatenate the datasets
    dataset = concatenate_datasets(datasets)
    dataset = dataset.shuffle(seed=cfg.seed)

    logger.info(dataset)

    # Load the prompt template
    prompt_template = open(cfg.generation.prompt_template_path, "r").read()

    # We apply chat template to each example in the dataset
    def apply_template(example):    
      tweet = example["text"]
      label = example["label"]

      messages = [
        {"role": "user", "content": prompt_template.format(tweet)},
      ]

      return {
        "prompt": messages,
        "answer": label
      }
      
    dataset: Dataset = dataset.map(apply_template, num_proc=4).skip(cfg.dataset.skip) # We do the skip mostly for checkpoint restarts
    dataset = dataset.remove_columns(['text', 'label'])

    def reward_fn(completions, answer, **kwargs):
      rewards = []
      completions = [completion[0]["content"] for completion in completions]
      label = answer
      for completion, l in zip(completions, label):
        classes = ["negative", "neutral", "positive"]
        wrapped_classes = ["boxed{negative}", "boxed{neutral}", "boxed{positive}"]
        in_classes = [c in completion for c in wrapped_classes]

        if sum(in_classes) != 1:
            rewards.append(0)
            continue

        index_pred = in_classes.index(True)
        index_true = classes.index(l)

        if index_pred == index_true:
            rewards.append(1)
        elif abs(index_pred - index_true) == 1:
            rewards.append(0.5)
        else:
            rewards.append(0)
        

        if len(completion) < 50:
            rewards[-1] -= 0.25
      return rewards
          

    trainer = CustomGRPOTrainer(
        model=model_config.model_name_or_path,
        reward_funcs=[reward_fn],
        args=GRPOConfig(
            gradient_accumulation_steps=cfg.train.number_of_problems_per_batch,
            gradient_checkpointing=train_config.gradient_checkpointing,
            num_generations=cfg.train.num_samples_per_problem,
            per_device_train_batch_size=cfg.train.num_samples_per_problem,
            # num_iterations=cfg.train.mu,

            # epsilon=cfg.train.epsilon,
            sync_ref_model=cfg.train.sync_ref_model,
            ref_model_mixup_alpha=cfg.train.ref_model_mixup_alpha,
            ref_model_sync_steps=cfg.train.ref_model_sync_steps,

            beta=cfg.train.beta,
            learning_rate=cfg.train.learning_rate,

            optim=cfg.train.optimizer,

            bf16=True,
            run_name=cfg.logging.wandb_run_name,
            model_init_kwargs=model_kwargs,
            hub_model_id=cfg.huggingface.name,
            hub_private_repo=False,
            report_to=["wandb"] if logging_config.wandb else [],
            save_strategy='steps',
            lr_scheduler_type=train_config.lr_scheduler_type,
            num_train_epochs=train_config.epochs,
            max_steps=train_config.max_steps,
            max_completion_length=model_config.max_length,
            logging_steps=1,
            save_steps=cfg.logging.save_steps,
            save_on_each_node=True,
            save_only_model=True,
            save_total_limit=3,
            output_dir=cfg.logging.save_dir,
            max_grad_norm=1.0,

            log_completions=False,
            use_vllm=True,
            vllm_device='cuda:0',
            vllm_gpu_memory_utilization=0.2,
            vllm_max_model_len=2000,
        ),
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    logger.info("Training...")
    train_results = trainer.train()
    logger.info("Training complete!")
    logger.info(train_results)

    kwargs = {
        "dataset_name": ','.join([dataset.name_or_path for dataset in cfg.dataset.datasets]),
        "tags": cfg.logging.wandb_tags,
    }

    if accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(logging_config.save_dir + "/model")
        trainer.model.save_pretrained(logging_config.save_dir + "/model")
        trainer.tokenizer.save_pretrained(logging_config.save_dir + "/model")

    if cfg.huggingface.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(private=True)
    

if __name__ == "__main__":
    main()
