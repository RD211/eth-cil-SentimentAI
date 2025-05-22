from functools import lru_cache
from datasets import DatasetDict
from omegaconf import DictConfig
from datasets import load_from_disk
from transformers import DataCollatorWithPadding

id2label = {0: "negative", 1: "neutral", 2: "positive"}
label2id = {v: k for k, v in id2label.items()}


@lru_cache(maxsize=128)
def get_prompt_template_text(path):
    with open(path) as f:
        return f.read()


def get_prompt(review: list[str], prompt_template_path: str, is_instruct: bool, tokenizer=None):
    prompt_template = get_prompt_template_text(prompt_template_path)
    prompt = prompt_template.format(review)

    if is_instruct:
        messages = [
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return text
    else:
        return prompt


def process_row(row, prompt_template_path: str, is_instruct: bool, tokenizer=None):
    try:
        review = row["text"]
        answer = row["label"]

        return {
            "text": get_prompt(review, prompt_template_path=prompt_template_path, is_instruct=is_instruct, tokenizer=tokenizer),
            "label": label2id[answer],
        }
    except Exception as e:
        review = row["text"]
        return {
            "text": get_prompt(review, prompt_template_path=prompt_template_path, is_instruct=is_instruct, tokenizer=tokenizer),
            "label": -1,
        }


def get_dataset(cfg: DictConfig, tokenizer=None):

    is_instruct = cfg.model.is_instruct
    data_config = cfg.data
    dataset = load_from_disk(data_config.path)

    # Process the dataset with the original prompt
    train_dataset = dataset.map(
        lambda row: process_row(row, data_config.prompt, is_instruct, tokenizer),
        num_proc=data_config.num_proc,
        desc="Processing",
    )

    dataset = DatasetDict({"train": train_dataset})

    if tokenizer is not None:
        dataset = dataset.map(
            lambda x: tokenizer(
                x["text"], truncation=True, max_length=data_config.max_seq_length
            ),
            batched=True,
            num_proc=data_config.num_proc,
            desc="Tokenizing",
        )

        collator = DataCollatorWithPadding(tokenizer=tokenizer)
        return dataset, collator

    return dataset

