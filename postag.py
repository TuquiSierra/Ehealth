import spacy
import pickle

POS_LIST = ["ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE"]
DEP_LIST = ["ROOT", "acl", "acomp", "advcl", "advmod", "agent", "amod", "appos", "attr", "aux", "auxpass", "case", "cc", "ccomp", "compound", "conj", "cop", "csubj", "csubjpass", "dative", "dep", "det", "dobj", "expl","expl:pass","fixed","flat",  "intj","iobj", "mark", "meta", "neg", "nn", "nmod","nummod", "nsubj", "nsubjpass", "oprd", "obj", "obl", "pcomp", "pobj", "poss", "preconj","parataxis", "prep", "prt", "punct",  "quantmod", "relcl", "root", "xcomp"]

def pos_tag(text):
    nlp=spacy.load('es_core_news_sm')
    doc =nlp(text)
    tags=[]
    for token in doc:
        tags.append([POS_LIST.index(token.pos_), DEP_LIST.index(token.dep_), text.index(token.head.text), POS_LIST.index(token.head.pos_)])

    return tags

def strip_punctuation(text):
    punc = '''!()[]{};:'"\,<>./?@#$%^&*_~'''
    for ele in punc:
        text = text.replace(ele, "") 
    return text


def pickle_postag(collection):
    postags={}
    for sentence in collection:
        postags[sentence.text]=pos_tag(strip_punctuation(" ".join(sentence.text.split())))
        
    filename=open('postag.data', 'wb')
    pickle.dump(postags,filename)
    filename.close()
    
        