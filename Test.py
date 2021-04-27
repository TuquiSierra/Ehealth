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
from functools import reduce
import pickle
from postag import pos_tag,pickle_postag



c = Collection()

c.load(Path("./2021/ref/training/medline.1200.es.txt"))
#         1      2      3     4      5      6     7      8      9     10     11    12      13    14     15     16     17
TAGS = ['B_C', 'I_C', 'L_C','B_A', 'I_A', 'L_A','B_P', 'I_P', 'L_P','B_R', 'I_R', 'L_R', 'U_C', 'U_A', 'U_P', 'U_R', 'V_C', 'V_A', 'V_P', 'V_R', 'O', 'V' ] 

LETTERS = string.printable + 'áéíóúÁÉÍÓÚñüö'


def letter_to_index(letter):
    return LETTERS.index(letter)

def line_to_tensor(line):
    tensor = torch.zeros(len(line), len(LETTERS))
    for i, c in enumerate(line):
        tensor[i][letter_to_index(c)] = 1
    
    return tensor

def label_to_tensor(label):
    return torch.tensor([TAGS.index(label)])


def sentence_to_tensor(sentence):
    words = sentence.split()
    sentence_len = len(words)
    words_representation = []
    for word in words:
        tensor = line_to_tensor(word)
        word_representation = (len(word), tensor)
        words_representation.append(word_representation)

    bert_vectors = bert_embeddings[sentence]
    postag_vectors=postags[sentence]
    #postag= pos_tag(strip_punctuation(" ".join(sentence.split())))
        
    return (sentence_len, words_representation, bert_vectors, postag_vectors)
    
def my_collate_fn(data):
    def reduce_fn(acum, value):
        return (acum[0] + [value[0]], acum[1] + [value[1]])
    samples, targets = reduce(reduce_fn, data, ([], []))
    targets = torch.cat(targets, dim=0).squeeze()
    return samples, targets


bert_embeddings = pickle.load(open('bert_embeddings_v2.data', 'rb'))
postags=pickle.load(open('postag.data', 'rb'))
criterion = nn.CrossEntropyLoss()
learning_rate = 0.005

def main():
    # c=Collection()
    # c.load(Path("./2021/ref/training/medline.1200.es.txt"))
    # pickle_postag(c)
    file = './2021/ref/training/medline.1200.es.txt'
    data = SentenceDataset(file, transform=sentence_to_tensor, target_transform=lambda l : torch.stack(tuple(map(label_to_tensor, l))))
    data_loader = DataLoader(data, batch_size=4, collate_fn=my_collate_fn, shuffle=True)
    n = MyLSTM(50, 50, len(TAGS), 113, 50 )
    optimizer = torch.optim.SGD(n.parameters(), lr=learning_rate)
    metrics = {
        'acc' : lambda pred, true : Accuracy()(pred, true),
        'f1' : lambda pred, true : F1Score()(torch.tensor(pred.argmax(dim=1), dtype=torch.float32), torch.tensor(true, dtype=torch.float32)) 
    }
    train(data_loader, n, criterion, optimizer, 5, filename='test_lstm.pth', metrics=metrics)

if __name__ == '__main__':
    main()
