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

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, input):
        hidden = self.initHidden(input.size(0))
        for letter in range(input.size()[1]):
            combined = torch.cat((input[:, letter], hidden), 1)
            hidden = self.i2h(combined)
            output = self.i2o(combined)
            output = self.softmax(output)
        return output
    
    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size) 

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
    
    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

def category_from_output(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return TAGS[category_i], category_i


# def train(rnn, category_tensor, line_tensor):
#     hidden = rnn.initHidden()
#     rnn.zero_grad()

#     for i in range(line_tensor.size()[0]):
#         output, hidden = rnn(line_tensor[i], hidden)

#     loss = criterion(output, category_tensor)
#     loss.backward()

#     for p in rnn.parameters():
#         p.data.add_(p.grad.data, alpha=-learning_rate)
    
#     return output, loss.item()

PLOT_EVERY = 100

def label_to_tensor(label):
    # tensor = torch.zeros(len(TAGS))
    # tensor[TAGS.index(label)] = 1
    return torch.tensor([TAGS.index(label)])
    # return tensor

def training(rnn, words_labels):
    current_loss = 0
    all_losses = []
    for i, (word, label) in enumerate(words_labels):
        output, loss = train(rnn, torch.tensor([TAGS.index(label)], dtype=torch.long), line_to_tensor(word))
        current_loss += loss
        if i % PLOT_EVERY == 0:
            all_losses.append(current_loss / PLOT_EVERY)
            current_loss = 0
    return all_losses

def evaluate(rnn, line_tensor):
    hidden = rnn.initHidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    return output

def predict(rnn, input_line):
    with torch.no_grad():
        output = evaluate(rnn, line_to_tensor(input_line))

        return category_from_output(output)
        # topv, topi = output.topk(1, 1, True)
        # return category_from_output()

# if __name__ == '__main__':
#     words, labels, sentences = get_words_with_tag(c)
#     n_hidden = 128
#     rnn = RNN(len(LETTERS), n_hidden, len(TAGS))
#     losses = training(rnn, zip(words[:-100], labels[:-100]))
#     plt.plot(losses)
#     plt.show()
    
#     while True:
#         index = int(input('Insert index: '))
#         category = predict(rnn, words[index])
#         print(f'From word: {words[index]} in sentence: {sentences[index]}')
#         print(f'Correct {labels[index]} getted {category[0]}')


def sentence_to_tensor(sentence):
    words = sentence.text.split()
    sentence_tensor = []
    for word in words:
        sentence_tensor.append(line_to_tensor(word))
    sentence_tensor.append(torch.rand(28, len(LETTERS)))
    sentence_tensor = pad_sequence(sentence_tensor, batch_first=True)
    sentence_tensor = sentence_tensor[:sentence_tensor.shape[0]-1]
    return sentence_tensor
    
# criterion = nn.NLLLoss()
criterion = nn.CrossEntropyLoss()
learning_rate = 0.005

if __name__ == '__main__':
    file = './2021/ref/training/medline.1200.es.txt'
    # rnn = RNN(len(LETTERS), 128, len(TAGS))
    # data = WordDataset('./2021/ref/training/medline.1200.es.txt', transform=line_to_tensor, target_transform=label_to_tensor)
    # batch_sampler = EqualLenghtSequence(data, 32)
    # dataLoader = DataLoader(data, batch_sampler=batch_sampler)
    # optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
    # metrics = {
    #     'acc' : lambda pred, true : Accuracy()(pred, true),
    #     'f1' : lambda pred, true : F1Score()(torch.tensor(pred.argmax(dim=1), dtype=torch.float32), torch.tensor(true, dtype=torch.float32)) 
    # }
    # train(dataLoader, rnn, criterion, optimizer, 5, filename='test.pth', metrics=metrics)

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



    # c = Collection()
    # c.load(Path(file))

    # sentence = c.sentences[11]
    # print(sentence.text)
    # cv = sentence_to_tensor(sentence)
    # print(cv.shape)
