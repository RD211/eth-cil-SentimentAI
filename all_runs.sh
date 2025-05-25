##############################################
# Train
##############################################

# Base models
python train_classifier.py --config-path=config/classifier/base --config-name "smollm2-1.7B"
python train_classifier.py --config-path=config/classifier/base --config-name "smollm2-360M"
python train_classifier.py --config-path=config/classifier/base --config-name "smollm2-135M"
python train_classifier.py --config-path=config/classifier/base --config-name "qwen3-0.6B"
python train_classifier.py --config-path=config/classifier/base --config-name "qwen3-1.7B"

# Instruct models
python train_classifier.py --config-path=config/classifier/instruct --config-name "smollm2-135M"
python train_classifier.py --config-path=config/classifier/instruct --config-name "smollm2-360M"
python train_classifier.py --config-path=config/classifier/instruct --config-name "qwen3-0.6B"
python train_classifier.py --config-path=config/classifier/instruct --config-name "smollm2-1.7B"
python train_classifier.py --config-path=config/classifier/instruct --config-name "qwen3-1.7B"

# Rag models
python train_classifier.py --config-path=config/classifier/rag --config-name "smollm2-135M"
python train_classifier.py --config-path=config/classifier/rag --config-name "smollm2-360M"
python train_classifier.py --config-path=config/classifier/rag --config-name "qwen3-0.6B"
python train_classifier.py --config-path=config/classifier/rag --config-name "smollm2-1.7B"
python train_classifier.py --config-path=config/classifier/rag --config-name "qwen3-1.7B"

python train_classifier.py --config-path=config/classifier/rag --config-name "smollm2-135M" run_name="SmolLM2-135M-Instruct-RAG" model.model_name="HuggingFaceTB/SmolLM2-135M-Instruct" +model.is_instruct=True
python train_classifier.py --config-path=config/classifier/rag --config-name "smollm2-360M" run_name="SmolLM2-360M-Instruct-RAG" model.model_name="HuggingFaceTB/SmolLM2-360M-Instruct" +model.is_instruct=True
python train_classifier.py --config-path=config/classifier/rag --config-name "smollm2-1.7B" run_name="SmolLM2-1.7B-Instruct-RAG" model.model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct" +model.is_instruct=True
python train_classifier.py --config-path=config/classifier/rag --config-name "qwen3-0.6B" run_name="Qwen3-0.6B-Instruct-RAG" model.model_name="Qwen/Qwen3-0.6B" +model.is_instruct=True
python train_classifier.py --config-path=config/classifier/rag --config-name "qwen3-1.7B" run_name="Qwen3-1.7B-Instruct-RAG" model.model_name="Qwen/Qwen3-1.7B" +model.is_instruct=True



##############################################
# Test
##############################################

# Unfinedtuned models
python test.py --models "HuggingFaceTB/SmolLM2-1.7B" --output_file "results/smollm2-1.7B-Unfinetuned.csv" --is_llm
python test.py --models "HuggingFaceTB/SmolLM2-360M" --output_file "results/smollm2-360M-Unfinetuned.csv" --is_llm
python test.py --models "HuggingFaceTB/SmolLM2-135M" --output_file "results/smollm2-135M-Unfinetuned.csv" --is_llm
python test.py --models "Qwen/Qwen3-0.6B-Base" --output_file "results/qwen3-0.6B-Unfinetuned.csv" --is_llm
python test.py --models "Qwen/Qwen3-1.7B-Base" --output_file "results/qwen3-1.7B-Unfinetuned.csv" --is_llm

python test.py --models "HuggingFaceTB/SmolLM2-1.7B-Instruct" --output_file "results/smollm2-1.7B-Instruct-Unfinetuned.csv" --is_llm --instruct
python test.py --models "HuggingFaceTB/SmolLM2-360M-Instruct" --output_file "results/smollm2-360M-Instruct-Unfinetuned.csv" --is_llm --instruct
python test.py --models "HuggingFaceTB/SmolLM2-135M-Instruct" --output_file "results/smollm2-135M-Instruct-Unfinetuned.csv" --is_llm --instruct
python test.py --models "Qwen/Qwen3-0.6B" --output_file "results/qwen3-0.6B-Instruct-Unfinetuned.csv" --is_llm --instruct
python test.py --models "Qwen/Qwen3-1.7B" --output_file "results/qwen3-1.7B-Instruct-Unfinetuned.csv" --is_llm --instruct



# Base models
python test.py --models "rd211/SmolLM2-1.7B" --output_file "results/smollm2-1.7B.csv"
python test.py --models "rd211/SmolLM2-360M" --output_file "results/smollm2-360M.csv"
python test.py --models "rd211/SmolLM2-135M" --output_file "results/smollm2-135M.csv"
python test.py --models "rd211/Qwen3-0.6B-Base" --output_file "results/qwen3-0.6B.csv"
python test.py --models "rd211/Qwen3-1.7B-Base" --output_file "results/qwen3-1.7B.csv"

# Instruct models
python test.py --models "rd211/SmolLM2-1.7B-Instruct" --output_file "results/smollm2-1.7B-instruct.csv" --instruct
python test.py --models "rd211/SmolLM2-360M-Instruct" --output_file "results/smollm2-360M-instruct.csv" --instruct
python test.py --models "rd211/Qwen3-0.6B-Instruct" --output_file "results/qwen3-0.6B-instruct.csv" --instruct
python test.py --models "rd211/SmolLM2-135M-Instruct" --output_file "results/smollm2-135M-instruct.csv" --instruct
python test.py --models "rd211/Qwen3-1.7B-Instruct" --output_file "results/qwen3-1.7B-instruct.csv" --instruct

# Rag models
python test.py --models "rd211/SmolLM2-1.7B-RAG" --output_file "results/smollm2-1.7B-rag.csv" --rag
python test.py --models "rd211/SmolLM2-360M-RAG" --output_file "results/smollm2-360M-rag.csv" --rag
python test.py --models "rd211/SmolLM2-135M-RAG" --output_file "results/smollm2-135M-rag.csv" --rag
python test.py --models "rd211/Qwen3-0.6B-Base-RAG" --output_file "results/qwen3-0.6B-rag.csv" --rag
python test.py --models "rd211/Qwen3-1.7B-Base-RAG" --output_file "results/qwen3-1.7B-rag.csv" --rag

python test.py --models "rd211/SmolLM2-1.7B-Instruct-RAG" --output_file "results/smollm2-1.7B-instruct-rag.csv" --rag --instruct
python test.py --models "rd211/SmolLM2-360M-Instruct-RAG" --output_file "results/smollm2-360M-instruct-rag.csv" --rag --instruct
python test.py --models "rd211/SmolLM2-135M-Instruct-RAG" --output_file "results/smollm2-135M-instruct-rag.csv" --rag --instruct
python test.py --models "rd211/Qwen3-0.6B-Instruct-RAG" --output_file "results/qwen3-0.6B-instruct-rag.csv" --rag --instruct
python test.py --models "rd211/Qwen3-1.7B-Instruct-RAG" --output_file "results/qwen3-1.7B-instruct-rag.csv" --rag --instruct

python test.py --models "HuggingFaceTB/SmolLM2-1.7B" --output_file "results/smollm2-1.7B-Unfinetuned-RAG.csv" --is_llm --rag
python test.py --models "HuggingFaceTB/SmolLM2-360M" --output_file "results/smollm2-360M-Unfinetuned-RAG.csv" --is_llm --rag
python test.py --models "HuggingFaceTB/SmolLM2-135M" --output_file "results/smollm2-135M-Unfinetuned-RAG.csv" --is_llm --rag
python test.py --models "Qwen/Qwen3-0.6B-Base" --output_file "results/qwen3-0.6B-Unfinetuned-RAG.csv" --is_llm --rag
python test.py --models "Qwen/Qwen3-1.7B-Base" --output_file "results/qwen3-1.7B-Unfinetuned-RAG.csv" --is_llm --rag