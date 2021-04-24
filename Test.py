from scripts.anntools import Collection
from pathlib import Path
from our_annotations import parse_sentence
import matplotlib.pyplot as plt
import numpy as np
import string
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from data import WordDataset, EqualLenghtSequence, chunk, SentenceDataset
from training import train
from torch.utils.data import DataLoader
from metrics import Accuracy, F1Score
from tagger import get_tag_list
from LSTMnn import MyLSTM

from bert import bert_embedding


c = Collection()

c.load(Path("./2021/ref/training/medline.1200.es.txt"))
#         1      2      3     4      5      6     7      8      9     10     11    12      13    14     15     16     17
TAGS = ['B_C', 'I_C', 'L_C','B_A', 'I_A', 'L_A','B_P', 'I_P', 'L_P','B_R', 'I_R', 'L_R', 'U_C', 'U_A', 'U_P', 'U_R', 'V_C', 'V_A', 'V_P', 'V_R', 'O', 'V' ] 

LETTERS = string.printable + 'áéíóúÁÉÍÓÚñüö'
def get_feature_vectors(collection):
    samples = list(map(lambda x : x.text, collection.sentences))
    characters = string.printable
    characters += 'áéíóúÁÉÍÓÚñüö'

    token_index = dict(zip(characters, range(1, len(characters) + 1)))
    max_length = 350
    results = np.zeros((len(samples), max_length))
    for i, sample in enumerate(samples):
        for j, character in enumerate(sample):
            index = token_index.get(character)
            if index is None:
                print(character)
            results[i, j] = index 
    return results

def get_labels(collection):
    samples = collection.sentences
    tag_index = dict(zip(TAGS, range(1, len(TAGS) + 1)))
    max_length = 60
    results = np.zeros((len(samples), max_length, max(tag_index.values()) + 1))
    for i, sample in enumerate(samples):
        for j, tag in enumerate(parse_sentence(sample)):
            tag = tag.split()[2]
            index = tag_index.get(tag)
            if index is None:
                print('HEREEEE')
                print(tag)
            results[i, j, index] = 1
    return results

def get_words_with_tag(collection):
    words, labels, sentences = [], [], []
    for sentence in collection.sentences:
        tags = parse_sentence(sentence)
        tags = map(lambda x : x.split()[2], tags)
        for word, tag in zip(sentence.text.split(), tags):
            words.append(word)
            labels.append(tag)
            sentences.append(sentence.text)
    return (words, labels, sentences)

def letter_to_index(letter):
    return LETTERS.index(letter)

def letter_to_tensor(letter):
    tensor = torch.zeros(1, len(LETTERS))
    tensor[0][letter_to_index(letter)] = 1
    return tensor

def line_to_tensor(line):
    tensor = torch.zeros(len(line), len(LETTERS))
    for i, c in enumerate(line):
        tensor[i][letter_to_index(c)] = 1
    
    return tensor

def label_to_tensor(label):
    return torch.tensor([TAGS.index(label)])

def sentence_to_tensor(sentence):
    words = sentence.text.split()
    sentence_tensor = []
    for i,word in enumerate(words):
        sentence_tensor.append(line_to_tensor(word))
    sentence_tensor.append(torch.rand(28, len(LETTERS)))
    sentence_tensor = pad_sequence(sentence_tensor, batch_first=True)
    sentence_tensor = sentence_tensor[:sentence_tensor.shape[0]-1]
    
    #appending bert
    bert=bert_embedding(sentence.text)
    final_sentence_tensor=torch.empty(sentence_tensor.shape[0], 28+ bert[0].shape[1],len(LETTERS))
    for i in range(len(words)):
        transp=torch.transpose(bert[i], 0, 1)
        zeroes=torch.zeros(1, len(LETTERS))
        zeroes[0][0]=1
        mult=torch.matmul(transp, zeroes)
        final_sentence_tensor[i] = torch.cat((sentence_tensor[i], mult), 0)
    
    return final_sentence_tensor
    
# criterion = nn.NLLLoss()
criterion = nn.CrossEntropyLoss()
learning_rate = 0.005

def main():
    file = './2021/ref/training/medline.1200.es.txt'
    data = SentenceDataset(file, transform=sentence_to_tensor, target_transform=lambda l : torch.stack(tuple(map(label_to_tensor, l))))
    
    batch_sampler = EqualLenghtSequence(data, 2)
    data_loader = DataLoader(data, batch_sampler=batch_sampler)
    n = MyLSTM(50, 50, len(TAGS), 113, 50, 50)
    optimizer = torch.optim.SGD(n.parameters(), lr=learning_rate)
    metrics = {
        'acc' : lambda pred, true : Accuracy()(pred, true),
        'f1' : lambda pred, true : F1Score()(torch.tensor(pred.argmax(dim=1), dtype=torch.float32), torch.tensor(true, dtype=torch.float32)) 
    }
    train(data_loader, n, criterion, optimizer, 5, filename='test_lstm.pth', metrics=metrics)

if __name__ == '__main__':
    main()
