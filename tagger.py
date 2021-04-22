
translate = {
    'Concept': 'C',
    'Action': 'A',
    'Reference': 'R',
    'Predicate': 'P'
}

def get_word_number_per_index(text):
    word_number_per_index = [0 for _ in range(len(text))]
    actual_word_number = 0
    previous_was_space = False
    for index, letter in enumerate(text):
        if letter == ' ':
            previous_was_space = True
        else:
            if previous_was_space:
                actual_word_number += 1
                previous_was_space = False
            word_number_per_index[index] = actual_word_number
    return word_number_per_index

def get_word_numbers(word_number_per_index, keyphrase):
    return [ word_number_per_index[word_segment[0]] for word_segment in keyphrase.spans ]

def get_tags(keyphrase_size, keyphrase):
    keyphrase_type = keyphrase.label
    suffix = translate[keyphrase_type]
    if keyphrase_size == 1:
        return [f'U_{suffix}']
    
    return [f'B_{suffix}'] + [f'I_{suffix}' for _ in range(keyphrase_size - 2)] + [f'L_{suffix}']

def get_tag_list(sentence):
    number_of_words = len(sentence.text.split())
    tag_list = ['O' for _ in range(number_of_words)]
    word_number_per_index = get_word_number_per_index(sentence.text)
    for keyphrase in sentence.keyphrases:
        word_numbers_in_keyphrase = get_word_numbers(word_number_per_index, keyphrase)
        tags = get_tags(len(word_numbers_in_keyphrase), keyphrase)
        for word_index, tag in zip(word_numbers_in_keyphrase, tags):
            if tag_list[word_index] != 'O':
                tag_list[word_index] = 'V'
            else:
                tag_list[word_index] = tag
    return tag_list





