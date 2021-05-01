from LSTMnn import MyLSTM
from utils import sentence_to_tensor
from torch.nn.functional import softmax
from scripts.anntools import Collection
from pathlib import Path
import torch
import pickle
import string

MODEL_PATH = "./r_entities_model.pth_20"
EMBEDDING_PATH = "./bert_embeddings_v2.data"
POSTAG_PATH = "postag.data"
file = './2021/ref/training/medline.1200.es.txt'


TAGS = [None, 'B_C', 'I_C', 'L_C','B_A', 'I_A', 'L_A','B_P', 'I_P', 'L_P','B_R', 'I_R', 'L_R', 'U_C', 'U_A', 'U_P', 'U_R', 'O', 'V' ] 
LETTERS = [ None ] + list(string.printable + 'áéíóúÁÉÍÓÚñüö')

def get_BILUOV_tags(sentence_str, model, embedding, postags):
    input = [sentence_to_tensor(sentence_str, embedding, postags)]
    print(input)
    output = model(input)
    output = softmax(output, dim=1)
    # print(output)
    output = output.argmax(dim = 1, keepdim=True)
    print(output)
    return [ TAGS[i] for i in output]



if __name__ == "__main__":
    c = Collection()
    model = MyLSTM(100, 100, len(TAGS), len(LETTERS), 100)
    d= torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(d)
    embedding = pickle.load(open(EMBEDDING_PATH, 'rb'))
    postags = pickle.load(open(POSTAG_PATH, 'rb'))
    c.load(Path(file))
    for i in range(1000): 
        s = c.sentences[i].text
    
        tags = get_BILUOV_tags(s, model, embedding, postags)
        print(tags)