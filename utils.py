from functools import reduce
import torch
import string


LETTERS = [ None ] + list(string.printable + 'áéíóúÁÉÍÓÚñüö')

def my_collate_fn(data):
    def reduce_fn(acum, value):
        return (acum[0] + [value[0]], acum[1] + [value[1]])
    samples, targets = reduce(reduce_fn, data, ([], []))
    targets = torch.cat(targets, dim=0).squeeze()
    return samples, targets

def sentence_to_tensor(sentence, bert_embeddings, postags):
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

def label_to_tensor(label, all_labels):
    return torch.tensor([all_labels.index(label)])

def letter_to_index(letter):
    return LETTERS.index(letter)

def line_to_tensor(line):
    tensor = torch.zeros(len(line), len(LETTERS))
    for i, c in enumerate(line):
        tensor[i][letter_to_index(c)] = 1
    return tensor

def get_weights(data_loader, labels):
    counted_labels = counting_labels(data_loader, labels)
    total_labels = sum(list(counted_labels.values()))
    weights = []
    for value in counted_labels.values():
        w = 0
        try:
            w = total_labels / value
        except:
            w = 0
        weights.append(w)
    weights = torch.tensor(weights)
    return weights

def counting_labels(data_loader, labels):
    count = { label : 0 for label in labels }
    for _, y in data_loader:
        for label in y:
            count[int(label)] += 1
    return count
