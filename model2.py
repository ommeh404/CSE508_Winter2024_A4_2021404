import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from sklearn.model_selection import train_test_split
import re
import os

class ReviewDataset(Dataset):
    def __init__(self, texts, summaries, tokenizer, max_length=512):
        self.texts = texts
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded_text = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        encoded_summary = self.tokenizer(
            self.summaries[idx],
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoded_text['input_ids'].squeeze(0),
            'attention_mask': encoded_text['attention_mask'].squeeze(0),
            'labels': encoded_summary['input_ids'].squeeze(0)
        }

def main():
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', padding_side='left')
    # Ensure the tokenizer recognizes the pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Initializing a new GPT-2 model...")
    config = GPT2Config.from_pretrained('gpt2')
    model = GPT2LMHeadModel(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Loading data...")
    data_path = 'C:\\Users\\ommeh\\Downloads\\IR-4\\archive (1)\\Reviews.csv'
    data = pd.read_csv(data_path)
    data = data[['Text', 'Summary']].dropna().reset_index(drop=True)
    data = data.sample(n=10000, random_state=42)
    data['Text'] = data['Text'].apply(lambda x: re.sub('<.*?>', '', x).lower().strip())
    data['Summary'] = data['Summary'].apply(lambda x: re.sub('<.*?>', '', x).lower().strip())

    train_texts, val_texts, train_summaries, val_summaries = train_test_split(
        data['Text'], data['Summary'], test_size=0.25, random_state=42)

    train_dataset = ReviewDataset(train_texts.tolist(), train_summaries.tolist(), tokenizer)
    val_dataset = ReviewDataset(val_texts.tolist(), val_summaries.tolist(), tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    print("Starting training...")
    optimizer = AdamW(model.parameters(), lr=5e-5)
    epochs = 3
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            inputs = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)} processed.")
        print(f"Epoch {epoch+1} completed | Average Loss: {total_loss / len(train_loader)}")

    # Save the model and tokenizer
    model_save_path = 'C:\\Users\\ommeh\\Downloads\\IR-4\\model_saved'
    tokenizer_save_path = 'C:\\Users\\ommeh\\Downloads\\IR-4\\tokenizer_saved'
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(tokenizer_save_path, exist_ok=True)
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(tokenizer_save_path)
    print(f"Model and tokenizer saved to {model_save_path} and {tokenizer_save_path} respectively.")

if __name__ == '__main__':
    main()
