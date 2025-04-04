python auto_gpu.py --run "accelerate launch --config_file deepspeed/1GPU.yaml train_rl.py --config-name 0.5b.yaml" --disk 300 --filter "gpu_name=RTX4090"



accelerate launch --config_file deepspeed/1GPU.yaml train_rl.py --config-name 0.5b.yaml
accelerate launch --config_file deepspeed/1GPU.yaml train_rl.py --config-name 1.5b.yaml
python train_classifier.py