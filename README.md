# 🤖 SentimentAI: Computational Intelligence Lab Project

This project, "SentimentAI," was developed for the Computational Intelligence Lab (FS2025) at ETH Zurich. It provides the code and instructions to reproduce experiments on sentiment classification using instruction-tuned language models (like Qwen3 and SmolLM2), fine-tuning, and Retrieval-Augmented Generation (RAG).

The primary goal is to classify text into "positive," "neutral," or "negative" sentiments.

## 📜 Table of Contents
1.  [🛠️ Setup and Installation](#setup-and-installation)
2.  [🚀 Running the Experiments](#running-the-experiments)
3.  [🤗 Hugging Face Hub Resources & Model Performance](#hugging-face-hub-resources--model-performance)
4.  [📂 File Structure](#file-structure)
5.  [✍️ Authors](#authors)

## 🛠️ Setup and Installation

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Environment Variables:**
    Create a `.env` file in the root directory of the project and add your API keys if you plan to use Weights & Biases or push models to Hugging Face:
    ```env
    HF_TOKEN=your_hugging_face_api_token_if_needed
    WANDB_API_KEY=your_weights_and_biases_api_key_if_needed
    ```

3.  **Split the Data:**
    This script prepares the training, validation, and test datasets from `data/training.csv` and `data/test.csv`.
    ```bash
    python split_data.py
    ```

## 🚀 Running the Experiments

Configuration for training is managed via Hydra. Refer to the `config/` directory and `all_runs.sh` for detailed examples.

1.  **Train Classifiers (`train_classifier.py`):**
    This script handles model training. Main configurations are in `config/classifier/`.
    The `all_runs.sh` script contains a comprehensive list of training commands.

    **Basic Training Examples:**
    *   Train a base model (e.g., SmolLM2-1.7B):
        ```bash
        python train_classifier.py --config-path=config/classifier/base --config-name "smollm2-1.7B"
        ```
    *   Train an instruction-tuned model (e.g., SmolLM2-1.7B-Instruct):
        ```bash
        python train_classifier.py --config-path=config/classifier/instruct --config-name "smollm2-1.7B"
        ```
    *   Train with RAG (example for SmolLM2-1.7B-Instruct):
        ```bash
        python train_classifier.py --config-path=config/classifier/rag --config-name "smollm2-1.7B" run_name="SmolLM2-1.7B-Instruct-RAG" model.model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct" +model.is_instruct=True
        ```
    *   For more examples (different models, sizes, head initialization), see `all_runs.sh`.

2.  **Test Models (`test.py`):**
    This script evaluates models. Results are saved in the `results/` directory.
    The `all_runs.sh` script also provides testing examples.

    **Basic Testing Examples:**
    *   Test an unfine-tuned base LLM from Hugging Face:
        ```bash
        python test.py --models "HuggingFaceTB/SmolLM2-1.7B" --output_file "results/smollm2-1.7B-Unfinetuned.csv" --is_llm
        ```
    *   Test a fine-tuned model (replace `your-hf-username/model-name` with your actual model path on the Hub or a local path):
        ```bash
        python test.py --models "your-hf-username/SmolLM2-1.7B-Finetuned" --output_file "results/smollm2-1.7B-Finetuned.csv"
        ```
    *   Test a fine-tuned instruction-tuned model with RAG:
        ```bash
        python test.py --models "your-hf-username/SmolLM2-1.7B-Instruct-RAG" --output_file "results/smollm2-1.7B-instruct-rag.csv" --rag --instruct
        ```

## 🤗 Hugging Face Hub Resources & Model Performance

This section provides links to the Hugging Face Hub for fine-tuned models produced by this project, alongside their performance scores from the paper.

### SmolLM2 Family

*   **135M Parameters**
    *   Base: 0.80317 - [rd211/SmolLM2-135M-Base](https://huggingface.co/rd211/SmolLM2-135M-Base)
    *   Instruct: 0.79330 - [rd211/SmolLM2-135M-Instruct](https://huggingface.co/rd211/SmolLM2-135M-Instruct)
    *   Instruct-RAG: 0.80621 - [rd211/SmolLM2-135M-Instruct-RAG](https://huggingface.co/rd211/SmolLM2-135M-Instruct-RAG)
*   **360M Parameters**
    *   Base: 0.83287 - [rd211/SmolLM2-360M-Base](https://huggingface.co/rd211/SmolLM2-360M-Base)
    *   Instruct: 0.82798 - [rd211/SmolLM2-360M-Instruct](https://huggingface.co/rd211/SmolLM2-360M-Instruct)
    *   Instruct-RAG: 0.82180 - [rd211/SmolLM2-360M-Instruct-RAG](https://huggingface.co/rd211/SmolLM2-360M-Instruct-RAG)
*   **1.7B Parameters**
    *   Base: 0.89263 - [rd211/SmolLM2-1.7B-Base](https://huggingface.co/rd211/SmolLM2-1.7B-Base)
    *   Instruct: 0.89798 - [rd211/SmolLM2-1.7B-Instruct](https://huggingface.co/rd211/SmolLM2-1.7B-Instruct)
    *   Instruct-RAG: **0.90057** - [rd211/SmolLM2-1.7B-Instruct-RAG](https://huggingface.co/rd211/SmolLM2-1.7B-Instruct-RAG) (Best Score)

### Qwen3 Family

*   **0.6B Parameters**
    *   Base: 0.85178 - [rd211/Qwen3-0.6B-Base](https://huggingface.co/rd211/Qwen3-0.6B-Base)
    *   Instruct: 0.84329 - [rd211/Qwen3-0.6B-Instruct](https://huggingface.co/rd211/Qwen3-0.6B-Instruct)
    *   Instruct-RAG: 0.84569 - [rd211/Qwen3-0.6B-Instruct-RAG](https://huggingface.co/rd211/Qwen3-0.6B-Instruct-RAG)
*   **1.7B Parameters**
    *   Base: 0.88756 - [rd211/Qwen3-1.7B-Base](https://huggingface.co/rd211/Qwen3-1.7B-Base)
    *   Instruct: 0.88646 - [rd211/Qwen3-1.7B-Instruct](https://huggingface.co/rd211/Qwen3-1.7B-Instruct)
    *   Instruct-RAG: 0.88470 - [rd211/Qwen3-1.7B-Instruct-RAG](https://huggingface.co/rd211/Qwen3-1.7B-Instruct-RAG)

## 📂 File Structure
```
.
├── all_runs.sh             # Script with all experiment commands
├── config/                 # Hydra configurations (base, instruct, rag)
├── data/                   # Raw (training.csv, test.csv) and processed data
├── prompt_templates/       # Prompt files (sentiment.txt, sentiment_kshot.txt)
├── .env                    # Environment variables (HF_TOKEN, WANDB_API_KEY)
├── requirements.txt        # Python dependencies
├── split_data.py           # Data splitting script
├── train_classifier.py     # Model training script
├── test.py                 # Model testing script
├── rag.py                  # RAG implementation
└── data_loader.py          # Data loading utilities
```

## ✍️ Authors

This project was a collaborative effort by:
*   Kaushik Karthikeyan
*   Sarah Verreault
*   Piotr Cichon
*   David Dinucu-Jianu

Correspondence:
*   Kaushik Karthikeyan: `kkarthikeyan@ethz.ch`
*   Sarah Verreault: `sverreault@ethz.ch`
*   Piotr Cichon: `pcichon@student.ethz.ch`
*   David Dinucu-Jianu: `ddinucu@student.ethz.ch`

Computational Intelligence Lab FS2025, ETH Zurich.
