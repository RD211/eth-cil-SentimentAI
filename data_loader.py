from rag import EmbeddingStore
from functools import lru_cache
from datasets import DatasetDict
from omegaconf import DictConfig
from datasets import load_from_disk
from transformers import DataCollatorWithPadding
from jinja2 import Template

id2label = {0: "negative", 1: "neutral", 2: "positive"}
label2id = {v: k for k, v in id2label.items()}


@lru_cache(maxsize=128)
def get_prompt_template_text(path):
    with open(path) as f:
        return f.read()
    
@lru_cache(maxsize=128)
def get_prompt_template(path):
    with open(path) as f:
        template = Template(f.read())
    return template


def get_prompt(review: list[str], prompt_template_path: str, is_instruct: bool, tokenizer=None, rag_examples=None):
    if rag_examples is not None:
        prompt_template = get_prompt_template(prompt_template_path)
        prompt = prompt_template.render(
            review=review,
            examples=rag_examples,
        )
    else:
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


def process_row(row, prompt_template_path: str, is_instruct: bool, tokenizer=None, rag_examples=None):
    try:
        review = row["text"]
        answer = row["label"]

        return {
            "text": get_prompt(review, prompt_template_path=prompt_template_path, is_instruct=is_instruct, tokenizer=tokenizer, rag_examples=rag_examples),
            "label": label2id[answer],
        }
    except Exception as e:
        review = row["text"]
        return {
            "text": get_prompt(review, prompt_template_path=prompt_template_path, is_instruct=is_instruct, tokenizer=tokenizer, rag_examples=rag_examples),
            "label": -1,
        }


def get_dataset(cfg: DictConfig, tokenizer=None, rag_dataset=None):

    is_instruct = cfg.model.is_instruct
    data_config = cfg.data
    dataset = load_from_disk(data_config.path)

    embedding_store = None
    rag_examples = None
    if cfg.rag.use_rag:
        print("Loading embedding store")

        # We split the dataset into two parts. rag_dataset and train_dataset
        if rag_dataset is None:
            dataset = dataset.train_test_split(test_size=cfg.rag.data_percentage_reserve, seed=cfg.data.seed)
            rag_dataset = dataset["test"]
            dataset = dataset["train"]
        embedding_store = EmbeddingStore(
            ds=rag_dataset,
            embedding_model=cfg.rag.embedding_model,
        )

        print("Embedding store loaded")

        # For each example in the dataset we get the k nearest neighbors
        rag_examples = []
        texts = dataset["text"]

        rag_examples = embedding_store.get_k_nearest_batched(
            queries=texts,
            k=cfg.rag.k,
            out_of=cfg.rag.out_of,
        )        

    # Process the dataset with the original prompt
    train_dataset = dataset.map(
        lambda row,idx: process_row(row, data_config.prompt, is_instruct, tokenizer, rag_examples=rag_examples[idx] if rag_examples is not None else None),
        num_proc=data_config.num_proc,
        desc="Processing",
        with_indices=True,
    )

    # print out the first 5 examples
    print("First 5 examples:")
    for i in range(5):
        print(f"Example {i}: {train_dataset[i]}")

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

