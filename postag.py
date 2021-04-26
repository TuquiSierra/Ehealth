import spacy

POS_LIST = ["ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE"]
DEP_LIST = ["ROOT", "acl", "acomp", "advcl", "advmod", "agent", "amod", "appos", "attr", "aux", "auxpass", "case", "cc", "ccomp", "compound", "conj", "cop", "csubj", "csubjpass", "dative", "dep", "det", "dobj", "expl","expl:pass","fixed","flat",  "intj","iobj", "mark", "meta", "neg", "nn", "nmod","nummod", "nsubj", "nsubjpass", "oprd", "obj", "obl", "pcomp", "pobj", "poss", "preconj", "prep", "prt", "punct",  "quantmod", "relcl", "root", "xcomp"]

def pos_tag(text):
    nlp=spacy.load('es_core_news_sm')
    doc =nlp(text)
    tags=[]
    for token in doc:
        tags.append([POS_LIST.index(token.pos_), DEP_LIST.index(token.dep_), text.index(token.head.text), POS_LIST.index(token.head.pos_)])

    return tags
