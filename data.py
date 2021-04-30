from pathlib import Path
from torch.utils.data import Dataset
from scripts.anntools import Collection
from tagger import get_tag_list
from torch.utils.data import Dataset, Sampler, BatchSampler
from torch import split, tensor
from random import shuffle, randint, choice
from functools import reduce


def strip_punctuation(text):
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for ele in punc:
        text = text.replace(ele, "") 
    return text

class WordDataset(Dataset):
    def __init__(self, file, transform=None, target_transform=None):
        collection = Collection()
        collection.load(Path(file))
        self.data = WordDataset.__get_words_with_tag(collection)
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        word, label = self.data[index]
        if self.transform:
            word = self.transform(word)
        if self.target_transform:
            label = self.target_transform(label)
        return  (word, label)

    @staticmethod 
    def __get_words_with_tag(collection):
        data = []
        for sentence in collection.sentences:
            tags = get_tag_list(sentence)
            for word, tag in zip(sentence.text.split(), tags):
                data.append((word, tag))
        return data


class SentenceDataset(Dataset):
    def __init__(self, file, transform=None, target_transform=None):
        collection = Collection()
        collection.load(Path(file))
        self.data = SentenceDataset.__get_sentences_and_tag_sequences(collection)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, tags = self.data[index]
        if self.transform:
            sentence = self.transform(sentence)
        if self.target_transform:
            tags = self.target_transform(tags)
        return sentence, tags
    
    @staticmethod
    def __get_sentences_and_tag_sequences(collection):
        data = []
        for sentence in collection.sentences:
            tags = get_tag_list(sentence)
            # sentence_text = strip_punctuation(sentence.text)
            sentence_text = sentence.text
            data.append((sentence_text, tags))
        return data


class EqualLenghtSequence(Sampler):
    def __init__(self, dataset, batch_size):
        self.samples_per_len = {}
        for i, (sample, _) in enumerate(dataset):
            try:
                self.samples_per_len[len(sample)].append(i)
            except KeyError:
                self.samples_per_len[len(sample)] = [i]
        
        for key, value in self.samples_per_len.items():
            shuffle(value)
            self.samples_per_len[key] = chunk(value, batch_size)
    
    def __iter__(self):
        batches = [ value for batch_list in self.samples_per_len.values() for value in batch_list ]
        shuffle(batches)
        return iter(batches)

    def __len__(self):
        return reduce(lambda x, y : x + y, map(len, self.samples_per_len.values()), 0)
        
def chunk(list_to_split, batch_size):
    result = []
    for i in range(0,len(list_to_split), batch_size):
        result.append(list_to_split[i : i + batch_size])
    return result


class RelationDataset(Dataset):
    def __init__(self, file, transform=None, target_transform=None):
        c = Collection()
        c.load(Path(file))
        relations = RelationDataset.__get_pair_of_words_with_relation(c)
        no_relations = RelationDataset.__get_no_related_entities(c, len(relations) // 10)
        no_relations = list(map(lambda x : (x, 'None'), no_relations))
        self.data = relations + no_relations
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, tags = self.data[index]
        if self.transform:
            sentence = self.transform(sentence)
        if self.target_transform:
            tags = self.target_transform(tags)
        return sentence, tags
    
    @staticmethod
    def __get_pair_of_words_with_relation(collection):
        relations = []
        for sentence in collection.sentences:
            for relation in sentence.relations:
                new_relation = ((relation.from_phrase.text, relation.to_phrase.text), relation.label)
                relations.append(new_relation)
        return relations
    
    @staticmethod
    def __get_no_related_entities(collection, N):
        no_related_entities = []
        number_of_sentences = len(collection.sentences)
        sentences_indexes = list(range(number_of_sentences))
        shuffle(sentences_indexes)
        for index in sentences_indexes:
            sentence = collection.sentences[index]
            relations = {(r.from_phrase, r.to_phrase) : 1 for r in sentence.relations}
            entities = list(map(lambda kp : kp.text, sentence.keyphrases))
            from_entity = choice(entities)
            to_entity = choice(entities)
            if not (from_entity, to_entity) in relations:
                no_related_entities.append((from_entity, to_entity))
            if len(no_related_entities) > N:
                break
        return no_related_entities

