from spacy.tokenizer import Tokenizer
import spacy
import pickle
import torch
from itertools import combinations

POS_LIST = ["ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE"]
DEP_LIST = ["ROOT", "acl", "acomp", "advcl", "advmod", "agent", "amod", "appos", "attr", "aux", "auxpass", "case", "cc", "ccomp", "compound", "conj", "cop", "csubj", "csubjpass", "dative", "dep", "det", "dobj", "expl","expl:pass","fixed","flat",  "intj","iobj", "mark", "meta", "neg", "nn", "nmod","nummod", "nsubj", "nsubjpass", "oprd", "obj", "obl", "pcomp", "pobj", "poss", "preconj","parataxis", "prep", "prt", "punct",  "quantmod", "relcl", "root", "xcomp"]

def pos_tag(text):
    nlp=spacy.load('es_core_news_sm')
    nlp.tokenizer = Tokenizer(nlp.vocab)
    doc =nlp(text)
    tags=[]
    for token in doc:
        onehot=torch.zeros(len(POS_LIST))
        onehot[POS_LIST.index(token.pos_)]=1
        tags.append(onehot)

    return tags

def pairs_postag(text):
    nlp=spacy.load('es_core_news_sm')
    nlp.tokenizer = Tokenizer(nlp.vocab)
    doc =nlp(text)
    return [d.vector for d in doc]
    

def strip_punctuation(text):
    punc = '''!()[]{};:'"\,<>./?@#$%^&*_~'''
    for ele in punc:
        text = text.replace(ele, "") 
    return text


def pickle_postag(collection, file_name):
    postags={}
    for sentence in collection:
        postags[sentence.text]=pos_tag(strip_punctuation(" ".join(sentence.text.split())))
        
    filename=open(f'{file_name}.data', 'wb')
    pickle.dump(postags,filename)
    filename.close()
    

        