#https://huggingface.co/docs/transformers/training
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
import numpy as np
import pandas as pd
import evaluate
from transformers import TrainingArguments, Trainer
from joblib import load
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(text):
    return tokenizer(text, padding="max_length", truncation=True)

train_page, train_label, train_content = load('sentence_train.joblib')
test_page, test_label, test_content = load('sentence_test.joblib')

class EncodedDataset():
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

tokenized_train = tokenize_function(train_content)
tokenized_test = tokenize_function(test_content)

train_dataset = EncodedDataset(tokenized_train, train_label)
eval_dataset = EncodedDataset(tokenized_test, test_label)

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()