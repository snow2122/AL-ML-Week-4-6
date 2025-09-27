import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# load the AG News dataset which has 4 categories: world, sports, business, sci/tech
dataset = load_dataset("ag_news")

# just printing the first sample to see what the data looks like
print(dataset["train"][0])

# tokenizer changes raw text into tokens (numbers) that BERT can understand
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    # apply tokenization to each text, pad/truncate so all sequences are the same length
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# tokenize the full dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# load BERT and set it up for classification with 4 possible labels
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)

# define training and testing splits
train_dataset = tokenized_datasets["train"]
test_dataset = tokenized_datasets["test"]

# function to calculate evaluation metrics after each training run
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}

# set up training arguments (batch size, epochs, learning rate, etc.)
training_args = TrainingArguments(
    output_dir="./results",              # where to save model outputs
    evaluation_strategy="epoch",         # evaluate at the end of each epoch
    learning_rate=2e-5,
    per_device_train_batch_size=16,      # training batch size
    per_device_eval_batch_size=16,       # evaluation batch size
    num_train_epochs=2,                  # small number of epochs for faster training
    weight_decay=0.01,                   # add weight decay for regularization
)

# create the Trainer object which handles training and evaluation
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# run the training process
trainer.train()

# run evaluation after training
results = trainer.evaluate()

# print out the results in a clear format
print("Model evaluation:")
print(f"Accuracy: {results['eval_accuracy']*100:.2f}%")
print(f"F1 Score: {results['eval_f1']:.4f}")

