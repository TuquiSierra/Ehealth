from data import SentenceDataset, RelationDataset
from training import train
from torch.utils.data import DataLoader
from LSTMnn import MyLSTM
from RelationTagger import RelationTagger
from utils import sentence_to_tensor, my_collate_fn, label_to_tensor, get_weights, pairs_to_tensor
from metrics import MyAccuracy, MyAccuracyAll, MyF1Score, MyPrecission, MyRecall
import torch.nn as nn
import string
import torch
import pickle
import re


def counting_labels(data_loader, labels):
    count = { label : 0 for label in labels }
    for _, y in data_loader:
        for label in y:
            count[int(label)] += 1
    return count


TAGS = [None, 'B_C', 'I_C', 'L_C','B_A', 'I_A', 'L_A','B_P', 'I_P', 'L_P','B_R', 'I_R', 'L_R', 'U_C', 'U_A', 'U_P', 'U_R', 'O', 'V' ] 
# TAGS = [None, 'None', 'is-a', 'same-as', 'has-property', 'part-of', 'causes', 'entails', 'in-time', 'in-place', 'in-context', 'subject', 'target', 'domain', 'arg']
LETTERS = [ None ] + list(string.printable + 'áéíóúÁÉÍÓÚñüö')
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


bert_embeddings = pickle.load(open('bert_embeddings_v2.data', 'rb'))
bert_embeddings_dev = pickle.load(open('bert_embeddings_dev_v2.data', 'rb'))
postags = pickle.load(open('postag.data', 'rb'))
postags_dev = pickle.load(open('postag_dev.data', 'rb'))
spacy_embeddings = pickle.load(open('spacy_embeddings.data', 'rb'))
criterion = nn.CrossEntropyLoss()
learning_rate = 0.005

class Output:
    # Types
    CONCEPT = 'Concept'
    ACTION = 'Action'
    PREDICATE = 'Predicate'
    REFERENCE = 'Reference'

    def __init__(self):
        self.T = []
        self.R = []

    def __str__(self):
        items = []
        for i, t in enumerate(self.T):
            items.append(f"T{i+1}\t{t[1][0]}\t{t[1][1]}\t{t[1][2]}")
        return '\n'.join(items)

    def add_T(self, i, typ, pos, words):
        self.T.append((i,(typ,pos, words)))

def translate_sentences(sentences):
    out = Output()

    def map_type(tag):
        if re.match('._C', tag):
            return Output().CONCEPT
        if re.match('._A', tag):
            return Output().ACTION
        if re.match('._P', tag):
            return Output().PREDICATE
        if re.match('._R', tag):
            return Output().REFERENCE

    index = 0
    for sentence in sentences[:5]:
        s = re.sub(' +', ' ', sentence[0])
        tags = sentence[1]
        s = s.split(' ')
        simple_tags = ''.join([re.sub('_.', '', t) for t in tags])
        start = [index]
        for i in range(1,len(s)):
            start.append(start[i-1] + len(s[i-1]) + 1)
            print(s[i-1])
            s[i-1] = re.sub(',|;|:|\.', '', s[i-1])
        index = start[-1] + len(s[-1]) + 1
        s[-1] = re.sub(',|;|:|\.', '', s[-1])
        
        # Simple entities
        for x in re.finditer('U', simple_tags):
            out.add_T(start[x.span()[0]], map_type(tags[x.span()[0]]), f'{start[x.span()[0]]} {start[x.span()[0]] + len(s[x.span()[0]])}', f'{s[x.span()[0]]}')
            # print(f'{start[x.span()[0]]} {start[x.span()[0]] + len(s[x.span()[0]])} {s[x.span()[0]]}')

        # Continuous entities
        for x in re.finditer('(B)(I*)(L)', simple_tags):
            positions = []
            words = []
            for i in range(x.span()[0],x.span()[1]):
                words.append(s[i])
                positions.append(f'{start[i]} {start[i] + len(s[i])}')
            out.add_T(start[x.span()[0]],map_type(tags[x.span()[0]]), ';'.join(positions), ' '.join(words))
            # print(f"{';'.join(positions)} {' '.join(words)}")

        # Discontinuos entities with V at start
        for x in re.finditer('(V+)((I*LO*)+)(I*L)', simple_tags):
            span = x.span()
            positions = [f'{start[span[0]]} {start[span[0]] + len(s[span[0]])}']
            words = [s[span[0]]]
            for i in range(span[0]+1,span[1]):
                if simple_tags[i] == 'L':
                    words.append(s[i])
                    positions.append(f'{start[i]} {start[i] + len(s[i])}')
                    out.add_T(start[x.span()[0]], map_type(tags[x.span()[0]]), ';'.join(positions), ' '.join(words))
                    # print(f"{';'.join(positions)} {' '.join(words)}")
                    positions = [f'{start[span[0]]} {start[span[0]] + len(s[span[0]])}']
                    words = [s[span[0]]]
                elif simple_tags[i] == 'I':
                    words.append(s[i])
                    positions.append(f'{start[i]} {start[i] + len(s[i])}')

        # Discontinuos entities with V at end
        for x in re.finditer('((BO)+)(B)(V+)', simple_tags):
            span = x.span()
            starts = []
            positions = []
            words = []
            for i in range(span[0]+1,span[1]):
                if simple_tags[i] == 'B':
                    starts.append((i, (f'{start[i]} {start[i] + len(s[i])}',s[i])))
                elif simple_tags[i] == 'V':
                    words.append(s[i])
                    positions.append(f'{start[i]} {start[i] + len(s[i])}')
            words = ' '.join(words)
            positions = ';'.join(positions)
            for i, s in starts:
                out.add_T(start[i], map_type(tags[i]), f'{s[1][0]};{positions}', f'{s[1][1]} {words}')
                # print(f'{s[0]};{positions} {s[1]} {words}')
    return out

if __name__ == '__main__':
    file = './2021/ref/training/medline.1200.es.txt'
    file_dev = './2021/eval/training/scenario1-main/output.txt'

    data = SentenceDataset(file, transform=lambda x : sentence_to_tensor(x, bert_embeddings, postags), target_transform=lambda l : torch.stack(tuple(map(lambda x: label_to_tensor(x, TAGS), l))))
    dev_data = SentenceDataset(file_dev, transform=lambda x : sentence_to_tensor(x, bert_embeddings_dev, postags_dev), target_transform=lambda l : torch.stack(tuple(map(lambda x: label_to_tensor(x, TAGS), l))))

    # print(translate_sentences(data.data))

    data_loader = DataLoader(data, batch_size=4, collate_fn=my_collate_fn, shuffle=True)
    dev_data_loader = DataLoader(dev_data, batch_size=4, collate_fn=my_collate_fn, shuffle=True)

    data_loader_to_count = DataLoader(data, batch_size=4, collate_fn=my_collate_fn, shuffle=True)
    weights = get_weights(data_loader_to_count, range(len(TAGS)))

    criterion = nn.CrossEntropyLoss(weight=weights)
    n = MyLSTM(100, 100, len(TAGS), len(LETTERS), 100 )
    n.to(DEVICE)
    # optimizer = torch.optim.SGD(n.parameters(), lr=learning_rate)
    # # torch.optim.RMSprop
    # # torch.optim.Adam
    optimizer = torch.optim.Adam(n.parameters(), lr = learning_rate)
    
    metrics = {
        'acc' : MyAccuracy,
        'acc2' : MyAccuracyAll,
        'precission' : MyPrecission,
        'recall' : MyRecall,
        'f1': MyF1Score
    }
    train(data_loader, n, criterion, optimizer, 10, filename='test_lstm.pth', validate=dev_data_loader, metrics=metrics)
    # rd = RelationDataset(file, transform=lambda x : pairs_to_tensor(x, spacy_embeddings), target_transform=lambda x : label_to_tensor(x, TAGS))
    # rdl = DataLoader(rd, batch_size=4, shuffle=True, collate_fn=my_collate_fn)

    # n = RelationTagger(96, len(TAGS))
    # optimizer = torch.optim.SGD(n.parameters(), lr=learning_rate)
    # train(rdl, n, criterion, optimizer, 10, filename='test_lstm.pth', metrics=metrics)
    # b = next(iter(rdl))
    # print(type(b))
    # print(len(b))
    # x = b[0]
    # print(type(x))
    # print(len(x))
    # s = x[0]
    # print(type(s))
    # print(len(s))
    # e = s[0]
    # print(type(e))
    # print(e.shape)
