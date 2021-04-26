from transformers import BertTokenizer, BertModel
import numpy as numpy
import torch
from scripts.anntools import Collection
from tagger import get_tag_list
import string
import torch.nn as nn


model =BertModel.from_pretrained('bert-base-multilingual-cased',from_tf=True,output_hidden_states=True,)
tokenizer =BertTokenizer.from_pretrained('bert-base-multilingual-cased')

def strip_punctuation(text):
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for ele in text: 
        if ele in punc: 
            text = text.replace(ele, "") 
    return text

def bert_text_preparation(text, tokenizer):
    marked_text = "[CLS] " + strip_punctuation(text) + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1]*len(indexed_tokens)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    return tokenized_text, tokens_tensor, segments_tensors


def get_bert_embeddings(tokens_tensor, segments_tensors, model):

    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2][1:]

    embeddings=torch.squeeze(hidden_states[-1])
    
    #adding up the last four hidden states
    for i in range(2,5):
        token_embeddings=hidden_states[-i]
        embeddings=[x+y for (x, y) in zip(embeddings, torch.squeeze(token_embeddings, dim=0))]
        
    list_token_embeddings = [token_embed.tolist() for token_embed in embeddings]

    return list_token_embeddings

def bert_embedding(text):
    tokenized_text, tokens_tensor, segment_tensors=bert_text_preparation(text, tokenizer)
    list_token_embeddings = get_bert_embeddings(tokens_tensor, segment_tensors, model) 
    i=1
    sentence_embedding=[]
    while i< len(tokenized_text)-1:
        embedding =list_token_embeddings[i][:178]
        i+=1
        while i< len(tokenized_text) and tokenized_text[i][0]=='#':
            embedding=[x+y for (x, y) in zip(embedding, list_token_embeddings[i])]    
            i+=1
        sentence_embedding.append(torch.FloatTensor(embedding).view(1, -1))

    return sentence_embedding
