## Needs
```bash
pip install -r requirements.txt
```
.env file with:
```
HF_TOKEN=xxx
WANDB_API_KEY=xxx
```

## Split the data
```bash
python split_data.py
```

## Train classifier
```bash
python train_classifier.py
```

## Validate and test
```bash
python test.py
```

## RL Training
```bash
accelerate launch --config_file deepspeed/1GPU.yaml train_rl.py --config-name 1.5b.yaml
```