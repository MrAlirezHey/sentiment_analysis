# sentiment_analysis
# Sentiment Analysis using BERT (IMDB Dataset)

This project demonstrates how to fine-tune a sentiment classifier using a frozen BERT backbone on the IMDB movie reviews dataset. Only the classification head is trained, making it a lightweight and efficient method for binary sentiment analysis.

## 💡 Overview

- Dataset: [IMDB Movie Reviews](https://huggingface.co/datasets/imdb)
- Model: `bert-base-uncased` from HuggingFace Transformers
- Framework: PyTorch
- Task: Binary Sentiment Classification (Positive / Negative)

---

## 📦 Installation

Install required packages:

```bash
pip install transformers datasets evaluate
pip install --upgrade datasets huggingface-hub fsspec

📁 Dataset & Tokenization
We use the HuggingFace Datasets library to load and tokenize the IMDB dataset:

from transformers import BertTokenizer
from datasets import load_dataset

dataset = load_dataset("imdb")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenizer_function(samples):
    return tokenizer(samples['text'], padding='max_length', truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenizer_function, batched=True)
tokenized_dataset = tokenized_dataset.rename_column('label', 'labels')
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
🧠 Model Architecture
The base BERT model is frozen. We train a custom classifier head on top:

import torch.nn as nn
from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self, bert_model):
        super(BertClassifier, self).__init__()
        self.bert = bert_model
        for param in self.bert.parameters():
            param.requires_grad = False
        self.fc1 = nn.Linear(768, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.relu(self.fc1(pooled_output))
        return self.fc2(x)
🏋️ Training
from torch.utils.data import DataLoader
import torch.optim as optim

train_dataloader = DataLoader(tokenized_dataset['train'], shuffle=True, batch_size=16)
test_dataloader = DataLoader(tokenized_dataset['test'], batch_size=32)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(model.fc1.parameters()) + list(model.fc2.parameters()), lr=5e-5)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    print(f"Epoch {epoch+1}: Loss={total_loss/len(train_dataloader):.4f}, Accuracy={correct/total:.4f}")
✅ Evaluation
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        logits = model(input_ids, attention_mask)
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

print(f"Validation Accuracy: {correct / total:.4f}")
🔍 Inference
def predict_sentiment(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_mask)

    prediction = logits.argmax(dim=1).item()
    sentiment = 'Positive' if prediction == 1 else 'Negative'
    print(f'Sentence: {sentiment}')
    return prediction

