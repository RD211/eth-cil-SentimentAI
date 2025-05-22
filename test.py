from omegaconf import OmegaConf
import tqdm
import os
import gc
import torch
import torch.nn as nn
import argparse
import numpy as np
import pandas as pd
from datasets import load_from_disk
from data_loader import get_dataset, id2label, label2id
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from hydra.core.config_store import ConfigStore
from config.train_classifier import Config as ClassifierConfig

cs = ConfigStore.instance()
cs.store(name="config", node=ClassifierConfig)


def run_ensemble_inference(dss, models, tokenizers, desc):
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm.tqdm(list(zip(*dss)), desc=desc):

            ensemble_logits = None
            for i in range(len(models)):
                inputs = {
                    "input_ids": batch[i]["input_ids"],
                    "attention_mask": batch[i]["attention_mask"],
                }
                model = models[i]
                batched_inputs = tokenizers[i].pad(inputs, return_tensors="pt")

                batched_inputs = {k: v.to("cuda") for k, v in batched_inputs.items()}
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

    parser = argparse.ArgumentParser(
        description="Validate and test a classification model with ensemble support."
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["rd211/SmolLM2-1.7b-base-NOheadinit-64bz-95data"],
        help="List of model names or paths for ensemble",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for inference"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="submission.csv",
        help="Output file for the submission",
    )
    parser.add_argument(
        "--instruct",
        action="store_true",
        help="Use chat template for inference",
    )
    parser.add_argument(
        "--is_llm",
        action="store_true",
        help="Use LLM model for inference by replacing head.",
    )

    args = parser.parse_args()

    model_names = args.models
    batch_size = args.batch_size

    # Load the validation and test datasets.
    val_dataset = load_from_disk("data/validation")

    # Load all models and a common tokenizer (using the first model's tokenizer)
    models = []
    for m in model_names:
        model = AutoModelForSequenceClassification.from_pretrained(
            m, 
            torch_dtype=torch.bfloat16, 
            device_map="auto", 
            num_labels=3, 
            label2id=label2id, 
            id2label=id2label
        )
        model.eval()
        model.config.use_cache = False
        models.append(model)

        tokenizer = AutoTokenizer.from_pretrained(m, trust_remote_code=True)
        if args.is_llm:
            print("Loading llm weights for head initialization.")
            llm = AutoModelForCausalLM.from_pretrained(
                m,
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


    tokenizers = [AutoTokenizer.from_pretrained(model_names[i]) for i in range(len(model_names))]
    # For each tokenizer we set pad token if not already set.
    for i in range(len(tokenizers)):
        tokenizers[i].pad_token = tokenizers[i].eos_token
        print(f"Tokenizer {model_names[i]} pad token set to eos token.")
        models[i].config.pad_token_id = tokenizers[i].pad_token_id

    # -----------------------
    # Validation Inference
    # -----------------------
    cfg = OmegaConf.structured(ClassifierConfig)

    cfg.data.path = "./data/validation"
    cfg.model.is_instruct = args.instruct
    
    ds_vals = []
    for i in range(len(model_names)):
        ds_val, _ = get_dataset(cfg, tokenizer=tokenizers[i])
        ds_val = ds_val["train"].batch(batch_size)
        ds_vals.append(ds_val)

    final_predictions = run_ensemble_inference(
        ds_vals, models, tokenizers, desc="Validation Inference"
    )
    verdicts = final_predictions.tolist()
    verdicts = [id2label[i] for i in verdicts]
    print("Validation sample verdicts:", verdicts[:10])

    # Scoring function for validation
    def score(label, verdict):
        if label == "positive":
            if verdict == "positive":
                return 1
            elif verdict == "neutral":
                return 0.5
            else:
                return 0
        elif label == "negative":
            if verdict == "negative":
                return 1
            elif verdict == "neutral":
                return 0.5
            else:
                return 0
        else:  # label is neutral
            if verdict == "neutral":
                return 1
            elif verdict == "positive":
                return 0.5
            else:
                return 0.5

    scores = []
    labels = val_dataset["label"]
    for i in range(len(labels)):
        scores.append(score(labels[i], verdicts[i]))
    mean_score = np.mean(scores)
    print("Validation mean score:", mean_score)

    # We save a file in results/model_name_validation.csv
    # With just the validation score
    model_name = model_names[0].split("/")[-1]
    results_path = f"results/{model_name}_validation.csv"
    if not os.path.exists("results"):
        os.makedirs("results")
    pd.DataFrame({"score": [mean_score]}).to_csv(results_path, index=False)

    # -----------------------
    # Test Inference & Submission
    # -----------------------
    cfg.data.path = "./data/test"
    cfg.model.is_instruct = args.instruct

    ds_tests = []

    for i in range(len(model_names)):
        ds_test, _ = get_dataset(cfg, tokenizer=tokenizers[i])
        ds_test = ds_test["train"].batch(batch_size)

        ds_tests.append(ds_test)

    final_predictions = run_ensemble_inference(
        ds_tests, models, tokenizers, desc="Test Inference"
    )
    verdicts = final_predictions.tolist()
    verdicts = [id2label[i] for i in verdicts]
    print("Test sample verdicts:", verdicts[:10])

    # Flatten the test IDs
    ids_ = ds_test["id"]
    ids = []
    for id_list in ids_:
        ids.extend(id_list)
    print("Test sample ids:", ids[:10])

    submission_df = pd.DataFrame({"id": ids, "label": verdicts})
    submission_df.to_csv(args.output_file, index=False)
    print("Submission saved to submission.csv")


if __name__ == "__main__":
    main()
