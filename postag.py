import spacy

def pos_tag(text):
    nlp=spacy.load('es_core_news_sm')
    doc =nlp(text)
    tags=[]
    for token in doc:
        tags.append([token.pos, token.dep, text.index(token.head.text), token.head.pos])
        
    print(tags)
    
    return tags