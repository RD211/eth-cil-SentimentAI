from datasets import Dataset
import pandas as pd

test_df = pd.read_csv('data/test.csv')
train_df = pd.read_csv('data/training.csv')

train_dataset = Dataset.from_dict({'text': train_df['sentence'], 'label': train_df['label']})
test_dataset = Dataset.from_dict({'id': test_df['id'], 'text': test_df['sentence']})
# shuffle
train_dataset = train_dataset.shuffle()

validation_dataset = train_dataset.train_test_split(test_size=0.05)
train_dataset = validation_dataset['train']
validation_dataset = validation_dataset['test']
# Save the datasets to disk
train_dataset.save_to_disk('data/train')
test_dataset.save_to_disk('data/test')
validation_dataset.save_to_disk('data/validation')