import torch
from torch import nn
from torch.nn.functional import softmax


BERT_VECTOR_SIZE = 178


class MyLSTM(nn.Module):
    def __init__(self, word_dimensions, main_hidden_size, output_size, number_of_letters, secondary_hidden_size):
        super(MyLSTM, self).__init__()
        self.word_dimensions = word_dimensions
        self.main_hidden_size = main_hidden_size
        self.output_size = output_size
        self.number_of_letters = number_of_letters
        self.secondary_hidden_size = secondary_hidden_size

        # Word representation netword
        self.word_lstm = nn.LSTM(input_size=self.number_of_letters,
                                 hidden_size=secondary_hidden_size, batch_first=True)

        self.word_linear = nn.Linear(
            secondary_hidden_size, word_dimensions)

        # Sentence analysis
        lstm_input_size = self.word_dimensions + BERT_VECTOR_SIZE
        self.sentence_lstm = nn.LSTM(
            input_size=lstm_input_size, hidden_size=self.main_hidden_size, batch_first=True, bidirectional=True)

        self.output_layer = nn.Linear(
            self.main_hidden_size*2, self.output_size)

    def __init_main_hidden(self, batch_size):
        hidden_state = torch.randn(2, batch_size, self.main_hidden_size)
        cell_state = torch.randn(2, batch_size, self.main_hidden_size)
        return (hidden_state, cell_state)

    def __init_secondary_hidden(self, batch_size):
        hidden_state = torch.randn(1, batch_size, self.secondary_hidden_size)
        cell_state = torch.randn(1, batch_size, self.secondary_hidden_size)
        return (hidden_state, cell_state)

    def forward(self, X):
        batch_size, sentence_length, *_ = X.shape

        # Go to word level
        number_of_words = sentence_length * batch_size
        X = X.view(number_of_words, -1, self.number_of_letters)

        # Separate word letters from BERT
        words_lenght = X.shape[1] - BERT_VECTOR_SIZE
        letters = X[:, :words_lenght, :]
        bert = self.__get_bert_vectors(X)

        # Words througth lstm
        hidden = self.__init_secondary_hidden(number_of_words)
        output, _ = self.word_lstm(letters, hidden)

        # Output from lstm througth linear to get word representations
        word_representation = output[:, -1, :].squeeze(1)
        word_representation = self.word_linear(word_representation)

        # Create ultimate vectors :)
        word_representation = torch.cat((word_representation, bert), dim=1)
        # Go to sentece level
        sentences = word_representation.view(batch_size, sentence_length, -1)

        # Sentences througth lstm
        hidden = self.__init_main_hidden(batch_size)
        output, _ = self.sentence_lstm(sentences, hidden)

        # Go to level word
        words = output.reshape(batch_size * sentence_length, -1)

        # Words througth last linear layer
        tags = self.output_layer(words)

        # Get tags probs
        tags_probs = softmax(tags, 1)

        # Go to sentence level
        # tags_probs = tags_probs.view(batch_size, sentence_length, -1)

        return tags_probs

    def __get_bert_vectors(self, X):
        padded_bert = X[:, -BERT_VECTOR_SIZE:, :]
        cleaned_bert = torch.zeros(padded_bert.shape[0], BERT_VECTOR_SIZE)
        for i, vector in enumerate(padded_bert):
            for j in range(BERT_VECTOR_SIZE):
                cleaned_bert[i][j] = vector[j, 0]
        return cleaned_bert
