import argparse
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import tqdm
from hydra import compose, initialize
from omegaconf import OmegaConf
from data_loader import get_dataset, id2label
import pandas as pd
import numpy as np

def run_ensemble_inference(ds, models, tokenizer, batch_size, desc):
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm.tqdm(ds, desc=desc):
            inputs = {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask']
            }
            batched_inputs = tokenizer.pad(inputs, return_tensors="pt")

            batched_inputs = {k: v.to('cuda') for k, v in batched_inputs.items()}
            ensemble_logits = None

            for model in models:
                outputs = model(**batched_inputs)
                logits = outputs.logits
                if ensemble_logits is None:
                    ensemble_logits = logits
                else:
                    ensemble_logits += logits

            ensemble_logits = ensemble_logits / len(models)

            predictions = torch.argmax(ensemble_logits, dim=-1)
            all_predictions.append(predictions.cpu())
    final_predictions = torch.cat(all_predictions)
    return final_predictions

def main():
    parser = argparse.ArgumentParser(description="Validate and test a classification model with ensemble support.")
    # Accept multiple models via a list of names/paths.
    parser.add_argument('--models', type=str, nargs='+', default=['rd211/1.5b-base-classifier-headinit-64bz-95data', 'rd211/1.5b-base-classifier-headinit-64bz-95data-rerun'],
                        help='List of model names or paths for ensemble')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    args = parser.parse_args()

    model_names = args.models
    batch_size = args.batch_size

    # Load the validation and test datasets.
    val_dataset = load_from_disk('data/validation')
    test_dataset = load_from_disk('data/test')

    # Load all models and a common tokenizer (using the first model's tokenizer)
    models = []
    for m in model_names:
        model = AutoModelForSequenceClassification.from_pretrained(
            m, torch_dtype=torch.bfloat16, device_map="auto", num_labels=3
        )
        model.eval()
        models.append(model)
    tokenizer = AutoTokenizer.from_pretrained(model_names[0])

    # -----------------------
    # Validation Inference
    # -----------------------
    with initialize(version_base=None, config_path="config/classifier", job_name="train_val"):
        cfg = compose(config_name="train")
    cfg.data.path = './data/validation'
    ds_val, collator = get_dataset(cfg, tokenizer=tokenizer)
    ds_val = ds_val['train'].batch(batch_size)

    final_predictions = run_ensemble_inference(ds_val, models, tokenizer, batch_size, desc="Validation Inference")
    verdicts = final_predictions.tolist()
    verdicts = [id2label[i] for i in verdicts]
    print("Validation sample verdicts:", verdicts[:10])

    # Scoring function for validation
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
        else:  # label is neutral
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
    print("Validation mean score:", mean_score)

    # -----------------------
    # Test Inference & Submission
    # -----------------------
    with initialize(version_base=None, config_path="config/classifier", job_name="train_test"):
        cfg = compose(config_name="train")
    cfg.data.path = './data/test'
    ds_test, collator = get_dataset(cfg, tokenizer=tokenizer)
    ds_test = ds_test['train'].batch(batch_size)

    final_predictions = run_ensemble_inference(ds_test, models, tokenizer, batch_size, desc="Test Inference")
    verdicts = final_predictions.tolist()
    verdicts = [id2label[i] for i in verdicts]
    print("Test sample verdicts:", verdicts[:10])

    # Flatten the test IDs
    ids_ = ds_test['id']
    ids = []
    for id_list in ids_:
        ids.extend(id_list)
    print("Test sample ids:", ids[:10])

    submission_df = pd.DataFrame({'id': ids, 'label': verdicts})
    submission_df.to_csv('submission.csv', index=False)
    print("Submission saved to submission.csv")

if __name__ == "__main__":
    main()
