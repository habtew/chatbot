import nltk

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    pass

a = ["Organize", "Organizes", "organizer"]
stem_word = [stem(w) for w in a]
print(a)
print(stem_word)