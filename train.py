import json
from nltk_utils import tokenize, stem, bag_of_words

# open trainging data

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
print(tags)