from data import SentenceDataset
from training import train
from torch.utils.data import DataLoader
from LSTMnn import MyLSTM
from utils import sentence_to_tensor, my_collate_fn, label_to_tensor, get_weights
from metrics import MyAccuracy, MyAccuracyAll, MyF1Score, MyPrecission, MyRecall
import torch.nn as nn
import string
import torch
import pickle


def counting_labels(data_loader, labels):
    count = { label : 0 for label in labels }
    for _, y in data_loader:
        for label in y:
            count[int(label)] += 1
    return count


TAGS = [None, 'B_C', 'I_C', 'L_C','B_A', 'I_A', 'L_A','B_P', 'I_P', 'L_P','B_R', 'I_R', 'L_R', 'U_C', 'U_A', 'U_P', 'U_R', 'O', 'V' ] 
LETTERS = [ None ] + list(string.printable + 'áéíóúÁÉÍÓÚñüö')
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


bert_embeddings = pickle.load(open('bert_embeddings_v2.data', 'rb'))
bert_embeddings_dev = pickle.load(open('bert_embeddings_dev_v2.data', 'rb'))
postags = pickle.load(open('postag.data', 'rb'))
postags_dev = pickle.load(open('postag_dev.data', 'rb'))
criterion = nn.CrossEntropyLoss()
learning_rate = 0.005

if __name__ == '__main__':
    file = './2021/ref/training/medline.1200.es.txt'
    file_dev = './2021/eval/training/scenario1-main/output.txt'

    data = SentenceDataset(file, transform=lambda x : sentence_to_tensor(x, bert_embeddings, postags), target_transform=lambda l : torch.stack(tuple(map(lambda x: label_to_tensor(x, TAGS), l))))
    dev_data = SentenceDataset(file_dev, transform=lambda x : sentence_to_tensor(x, bert_embeddings_dev, postags_dev), target_transform=lambda l : torch.stack(tuple(map(lambda x: label_to_tensor(x, TAGS), l))))

    data_loader = DataLoader(data, batch_size=4, collate_fn=my_collate_fn, shuffle=True)
    dev_data_loader = DataLoader(dev_data, batch_size=4, collate_fn=my_collate_fn, shuffle=True)

    data_loader_to_count = DataLoader(data, batch_size=4, collate_fn=my_collate_fn, shuffle=True)
    weights = get_weights(data_loader_to_count, range(len(TAGS)))

    criterion = nn.CrossEntropyLoss(weight=weights)
    n = MyLSTM(100, 100, len(TAGS), len(LETTERS), 100 )
    n.to(DEVICE)
    optimizer = torch.optim.SGD(n.parameters(), lr=learning_rate)
    torch.optim.RMSprop
    torch.optim.Adam
    
    metrics = {
        'acc' : MyAccuracy,
        'acc2' : MyAccuracyAll,
        'precission' : MyPrecission,
        'recall' : MyRecall,
        'f1': MyF1Score
    }
    train(data_loader, n, criterion, optimizer, 10, filename='test_lstm.pth', validate=dev_data_loader, metrics=metrics)