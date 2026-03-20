"""
Shared model definition and helper functions.
Used by both train.py and app.py.
"""

import re
import torch
import torch.nn as nn


# --- Text Processing ---

def tokenize(text):
    """
        Lowercase, remove punctuation, split into words.
        Args:
            text: Input string

        Returns:
            List of tokens
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def build_vocab(texts):
    """
        Build word-to-index mapping from a list of text strings.

        Args:
            texts: List of text strings

        Returns:
            word2idx: Dict mapping word to index
    """
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for text in texts:
        for word in tokenize(text):
            if word not in word2idx:
                word2idx[word] = len(word2idx)
    return word2idx


def encode_text(text, word2idx, max_len=32):
    """
    Convert text to index sequence.

    Args:
        text: Input string
        word2idx: Vocabulary mapping
        max_len: Max sequence length

    Returns:
        List of token indices (padded)
    """
    tokens = tokenize(text)
    ids = [word2idx.get(t, 1) for t in tokens]  # 1 = UNK
    ids = ids[:max_len]                          # truncate
    ids += [0] * (max_len - len(ids))            # pad
    return ids


# --- LSTM Intent Classifier ---

class IntentClassifier(nn.Module):
    """LSTM-based intent classifier."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        """
            Initialize model.

            Args:
                vocab_size: Vocabulary size
                embed_dim: Embedding dimension
                hidden_dim: LSTM hidden size
                num_classes: Number of intent labels
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        """
            Forward pass.

            Args:
                x: Input tensor (batch, seq_len)

            Returns:
                Logits tensor
        """
        emb = self.dropout(self.embedding(x))
        _, (hidden, _) = self.lstm(emb)
        # concat forward and backward hidden states
        hidden = torch.cat((hidden[0], hidden[1]), dim=1)
        hidden = self.dropout(hidden)
        return self.fc(hidden)
