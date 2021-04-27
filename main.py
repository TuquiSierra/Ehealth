from data import SentenceDataset
from training import train
from torch.utils.data import DataLoader
from LSTMnn import MyLSTM
from utils import sentence_to_tensor, my_collate_fn, label_to_tensor
from metrics import Accuracy, F1Score
import torch.nn as nn
import string
import torch
import pickle


TAGS = ['B_C', 'I_C', 'L_C','B_A', 'I_A', 'L_A','B_P', 'I_P', 'L_P','B_R', 'I_R', 'L_R', 'U_C', 'U_A', 'U_P', 'U_R', 'V_C', 'V_A', 'V_P', 'V_R', 'O', 'V' ] 
LETTERS = string.printable + 'áéíóúÁÉÍÓÚñüö'
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


bert_embeddings = pickle.load(open('bert_embeddings_v2.data', 'rb'))
postags=pickle.load(open('postag.data', 'rb'))
criterion = nn.CrossEntropyLoss()
learning_rate = 0.005

if __name__ == '__main__':
    file = './2021/ref/training/medline.1200.es.txt'
    data = SentenceDataset(file, transform=lambda x : sentence_to_tensor(x, bert_embeddings, postags), target_transform=lambda l : torch.stack(tuple(map(lambda x: label_to_tensor(x, TAGS), l))))
    data_loader = DataLoader(data, batch_size=4, collate_fn=my_collate_fn, shuffle=True)
    n = MyLSTM(50, 50, len(TAGS), 113, 50 )
    n.to(DEVICE)
    optimizer = torch.optim.SGD(n.parameters(), lr=learning_rate)
    metrics = {
        'acc' : lambda pred, true : Accuracy()(pred, true),
        'f1' : lambda pred, true : F1Score()(torch.tensor(pred.argmax(dim=1), dtype=torch.float32), torch.tensor(true, dtype=torch.float32)) 
    }
    train(data_loader, n, criterion, optimizer, 5, filename='test_lstm.pth', metrics=metrics)