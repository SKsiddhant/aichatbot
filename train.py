import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

IGNORE_WORDS = ['?', '.', '!']


class ChatDataset(Dataset):
    def __init__(self, x, y):
        self.x_data = x
        self.y_data = y

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.x_data)


def build_training_data(intents):
    all_words, tags, xy = [], [], []
    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            w = tokenize(pattern)
            all_words.extend(w)
            xy.append((w, tag))

    all_words = sorted({stem(w) for w in all_words if w not in IGNORE_WORDS})
    tags = sorted(set(tags))

    X_train = np.array([bag_of_words(sentence, all_words) for sentence, _ in xy])
    y_train = np.array([tags.index(tag) for _, tag in xy])
    return all_words, tags, X_train, y_train


def train(intents_path='intents.json', out_path='data.pth', num_epochs=1000,
          batch_size=8, learning_rate=0.001, hidden_size=8, seed=0, verbose=True):
    with open(intents_path, 'r', encoding='utf-8') as f:
        intents = json.load(f)

    torch.manual_seed(seed)
    all_words, tags, X_train, y_train = build_training_data(intents)
    if verbose:
        print(len(X_train), "patterns,", len(tags), "tags,", len(all_words), "unique stemmed words")

    input_size = len(X_train[0])
    output_size = len(tags)
    loader = DataLoader(ChatDataset(X_train, y_train), batch_size=batch_size, shuffle=True, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss = None
    for epoch in range(num_epochs):
        for words, labels in loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)
            loss = criterion(model(words), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if verbose and (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "all_words": all_words,
        "tags": tags,
    }
    torch.save(data, out_path)
    if verbose:
        print(f'final loss: {loss.item():.4f}\ntraining complete. file saved to {out_path}')
    return out_path


if __name__ == "__main__":
    train()
