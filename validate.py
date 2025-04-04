from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader
import tqdm
from hydra import compose, initialize
from omegaconf import OmegaConf
from data_loader import get_dataset


MODEL = 'rd211/1.5b-classifier-headinit-64bz'
val_dataset = load_from_disk('data/validation')
test_dataset = load_from_disk('data/test')



model = AutoModelForSequenceClassification.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto", num_labels=3)
tokenizer = AutoTokenizer.from_pretrained(MODEL)



with initialize(version_base=None, config_path="config/classifier", job_name="train"):
    cfg = compose(config_name="train")


cfg.data.path = './data/validation'
ds_val, collator = get_dataset(cfg, tokenizer=tokenizer)

batch_size = 32
ds_val = ds_val['train'].batch(batch_size)

model.eval()

all_predictions = []
all_logits = []


with torch.no_grad():
    for batch in tqdm.tqdm(ds_val, desc="Inference"):
        inputs = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
        batched_inputs = tokenizer.pad(inputs, return_tensors="pt")
        batched_inputs = {k: v.to('cuda') for k, v in batched_inputs.items()}                
        # Forward pass through the model.
        outputs = model(**batched_inputs)
        logits = outputs.logits
        
        # Compute predictions using argmax (for metrics, not for differentiable loss).
        predictions = torch.argmax(logits, dim=-1)
        
        all_predictions.append(predictions.cpu())
        all_logits.append(logits.cpu())

final_predictions = torch.cat(all_predictions)
final_logits = torch.cat(all_logits)

from data_loader import id2label
verdicts = final_predictions.tolist()
verdicts = [id2label[i] for i in verdicts]
print(verdicts[:10])

import numpy as np
def score(label, verdict):
    if label == 'positive':
        if verdict == 'positive':
            return 1
        elif verdict == 'neutral':
            return 0.5
        else:
            return 0
    elif label == 'negative':
        if verdict == 'negative':
            return 1
        elif verdict == 'neutral':
            return 0.5
        else:
            return 0
    else:
        if verdict == 'neutral':
            return 1
        elif verdict == 'positive':
            return 0.5
        else:
            return 0.5
        
scores = []
labels = val_dataset['label']
for i in range(len(labels)):
    scores.append(score(labels[i], verdicts[i]))

mean_score = np.mean(scores)
print(mean_score)