from scripts.anntools import Collection
from pathlib import Path
import os

# c = Collection()

# c.load(Path("./2021/ref/training/medline.1200.es.txt"))

translate = {
    'Concept': 'C',
    'Action': 'A',
    'Reference': 'R',
    'Predicate': 'P'
}

def parse_sentence(s):
    text = s.text
    if text[-1] == '.':
        text = text[:-1]
    words = text.split(' ')
    acum = 0
    kpi = 0
    ckp = s.keyphrases[kpi]
    i =0
    response = []

    while i < len(words):
        if acum < ckp.spans[0][0]:
            w = words[i]
            i += 1
            response.append(f'{acum} {acum + len(w)}\tO\t{w}')
            acum += len(w) + 1
        elif ckp.spans[0][0] <= acum < ckp.spans[-1][-1]:
            w = words[i]
            i += 1
            for j in range(len(ckp.spans)):
                if acum == ckp.spans[j][0]:
                    if kpi < len(s.keyphrases) - 1 and ckp.spans[j] in s.keyphrases[kpi+1].spans:
                        response.append(f'{acum} {acum + len(w)}\tV_{translate[ckp.label]}\t{w}')
                        acum += len(w) + 1
                    elif len(ckp.spans) == 1:
                        response.append(f'{acum} {acum + len(w)}\tU_{translate[ckp.label]}\t{w}')
                        acum += len(w) + 1
                    elif j == 0:
                        response.append(f'{acum} {acum + len(w)}\tB_{translate[ckp.label]}\t{w}')
                        acum += len(w) + 1
                    elif j == len(ckp.spans) - 1:
                        response.append(f'{acum} {acum + len(w)}\tL_{translate[ckp.label]}\t{w}')
                        acum += len(w) + 1
                    else:
                        response.append(f'{acum} {acum + len(w)}\tI_{translate[ckp.label]}\t{w}')
                        acum += len(w) + 1
                    break
        elif kpi < len(s.keyphrases) - 1:
            kpi += 1
            ckp = s.keyphrases[kpi]
        else:
            w = words[i]
            i += 1
            response.append(f'{acum} {acum + len(w)}\tO\t{w}')
            acum += len(w) + 1
    return response

# for t in parse_sentence(c.sentences[894]):
#     print(t)