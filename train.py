import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np
# open trainging data
# import torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    # print(intent['patterns'])
    for pattern in intent['patterns']:
        tokenize_pattern = tokenize(pattern)
        # print(tokenize_pattern)
        all_words.extend(tokenize_pattern)
        xy.append((tokenize_pattern, tag))
ignore_words = ['?', '.', ',', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# separate data
x_train = []
y_train = []
for (tokenized_sentence, tag) in xy:
    bag = bag_of_words(tokenized_sentence, all_words)
    x_train.append(bag)

    label = tags.index(tag)
    y_train.append(label) #crossentropy lebel

X_train = np.array(x_train)
Y_train = np.array(y_train)


class ChatDataSet(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train

    
    # to access dataset with index
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples
    
# hyper parameters
batch_size = 8
dataset = ChatDataSet()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)