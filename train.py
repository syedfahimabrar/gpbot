"""
Train intent classification model.

Loads data, builds vocab, splits dataset,
trains and evaluates the model, and saves
artifacts for inference.
"""

import json
import random
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import IntentClassifier, build_vocab, encode_text


# --- Dataset ---

class IntentDataset(Dataset):
    """Dataset for intent classification."""
    def __init__(self, data, word2idx, label2idx, max_len=32):
        """
            Initialize dataset.

            Args:
                data: List of text-intent samples
                word2idx: Word to index mapping
                label2idx: Intent to index mapping
                max_len: Max token length
        """
        self.data = data
        self.word2idx = word2idx
        self.label2idx = label2idx
        self.max_len = max_len

    def __len__(self):
        """Return number of samples."""
        return len(self.data)

    def __getitem__(self, idx):
        """
            Get one sample.

            Args:
                idx: Sample index

            Returns:
                x: Encoded text tensor
                y: Label tensor
        """
        item = self.data[idx]
        x = torch.tensor(encode_text(item["text"], self.word2idx, self.max_len), dtype=torch.long)
        y = torch.tensor(self.label2idx[item["intent"]], dtype=torch.long)
        return x, y


# --- Train/Test Split (stratified) ---

def split_data(data, test_ratio=0.2):
    """
        Split data into train/test sets (stratified).

        Args:
            data: List of samples
            test_ratio: Fraction for test set

        Returns:
            train: Training samples
            test: Test samples
    """
    random.seed(42)
    by_intent = {}
    for item in data:
        by_intent.setdefault(item["intent"], []).append(item)

    train, test = [], []
    for items in by_intent.values():
        random.shuffle(items)
        split = int(len(items) * (1 - test_ratio))
        train.extend(items[:split])
        test.extend(items[split:])

    random.shuffle(train)
    random.shuffle(test)
    return train, test


# --- Main ---

if __name__ == "__main__":
    # config
    EMBED_DIM = 128
    HIDDEN_DIM = 128
    MAX_LEN = 32
    BATCH_SIZE = 64
    LR = 0.001
    EPOCHS = 30

    # load data
    with open("data/intent_data.json") as f:
        all_data = json.load(f)

    labels = sorted(set(item["intent"] for item in all_data))
    label2idx = {l: i for i, l in enumerate(labels)}
    idx2label = {i: l for l, i in label2idx.items()}

    train_data, test_data = split_data(all_data)
    word2idx = build_vocab([item["text"] for item in train_data])

    print(f"Data: {len(train_data)} train, {len(test_data)} test, {len(labels)} intents, {len(word2idx)} vocab")

    # dataloaders
    train_loader = DataLoader(IntentDataset(train_data, word2idx, label2idx, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(IntentDataset(test_data, word2idx, label2idx, MAX_LEN), batch_size=BATCH_SIZE)

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IntentClassifier(len(word2idx), EMBED_DIM, HIDDEN_DIM, len(labels)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # training loop
    best_acc = 0.0
    for epoch in range(EPOCHS):
        # train
        model.train()
        total_loss, correct, total = 0, 0, 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (output.argmax(1) == batch_y).sum().item()
            total += batch_y.size(0)

        # test
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x).argmax(1)
                test_correct += (preds == batch_y).sum().item()
                test_total += batch_y.size(0)

        train_acc = correct / total
        test_acc = test_correct / test_total
        print(f"Epoch {epoch+1:2d}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Train: {train_acc:.4f} | Test: {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/best_model.pt")

    print(f"\nBest test accuracy: {best_acc:.4f}")

    # save everything needed for inference
    with open("models/word2idx.json", "w") as f:
        json.dump(word2idx, f)
    with open("models/idx2label.json", "w") as f:
        json.dump(idx2label, f)
    with open("models/config.json", "w") as f:
        json.dump({"embed_dim": EMBED_DIM, "hidden_dim": HIDDEN_DIM, "max_len": MAX_LEN, "num_classes": len(labels)}, f)

    print("Saved to models/")
