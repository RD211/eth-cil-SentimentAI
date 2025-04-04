import hydra
from omegaconf import DictConfig
from datasets import load_dataset, DatasetDict, concatenate_datasets
from jinja2 import Template
from functools import lru_cache
from transformers import AutoTokenizer
from dotenv import load_dotenv
from transformers import DataCollatorWithPadding
from  datasets import load_from_disk
import os

id2label = {
    0: 'negative',
    1: 'neutral',
    2: 'positive'
}
label2id = {v: k for k, v in id2label.items()}

@lru_cache(maxsize=128)
def get_prompt_template_text(path):
    with open(path) as f:
        return f.read()

def get_prompt(review: list[str], prompt_template_path: str):
    prompt_template = get_prompt_template_text(prompt_template_path)
    prompt = prompt_template.format(review)
    return prompt


def process_row(row,  prompt_template_path: str):
    try:
        review = row['text']
        answer = row['label']

        return {
            "text": get_prompt(review, 
                            prompt_template_path=prompt_template_path),
            "label": label2id[answer]
        }
    except Exception as e:
        review = row['text']
        return {
            "text": get_prompt(review, 
                            prompt_template_path=prompt_template_path),
            "label": -1
        }

def get_dataset(cfg: DictConfig, tokenizer = None):
    data_config = cfg.data
    dataset = load_from_disk(data_config.path)
    # dataset = dataset.shuffle(seed=data_config.seed)
    
    # if data_config.test_size > 0.0:
    #     dataset = dataset.train_test_split(test_size=data_config.test_size)


    # Process the dataset with the original prompt
    train_dataset = dataset.map(lambda row: process_row(row, data_config.prompt), 
                                         num_proc=data_config.num_proc, 
                                        #  cache_file_name=data_config.cache_dir + '/train_preprocess.arrow', 
                                         desc='Processing')
    

    dataset = DatasetDict({'train': train_dataset})
    
    if tokenizer is not None:
        dataset = dataset.map(lambda x: tokenizer(x['text'], truncation=True, max_length=data_config.max_seq_length), batched=True, num_proc=data_config.num_proc, desc='Tokenizing')

        collator = DataCollatorWithPadding(tokenizer=tokenizer)
        return dataset, collator

    return dataset
