import torch
from torch import nn
from torch.nn.functional import softmax
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

BERT_VECTOR_SIZE = 768
POS_SIZE=4


class MyLSTM(nn.Module):
    def __init__(self, word_dimensions, main_hidden_size, output_size, number_of_letters, secondary_hidden_size):
        print(DEVICE)
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
        lstm_input_size = self.word_dimensions + BERT_VECTOR_SIZE + POS_SIZE
        self.sentence_lstm = nn.LSTM(
            input_size=lstm_input_size, hidden_size=self.main_hidden_size, batch_first=True, bidirectional=True)

        self.output_layer = nn.Linear(
            self.main_hidden_size*2, self.output_size)

    def __init_main_hidden(self, batch_size):
        hidden_state = torch.randn(2, batch_size, self.main_hidden_size)
        cell_state = torch.randn(2, batch_size, self.main_hidden_size)
        return (hidden_state.to(DEVICE), cell_state.to(DEVICE))

    def __init_secondary_hidden(self, batch_size):
        hidden_state = torch.randn(1, batch_size, self.secondary_hidden_size)
        cell_state = torch.randn(1, batch_size, self.secondary_hidden_size)
        return (hidden_state.to(DEVICE), cell_state.to(DEVICE))

    def forward(self, X):
        batch_size = len(X)
        sentences_sizes = list(map(lambda sample: sample[0], X))
        word_sizes = []
        word_tensors = []
        bert_vectors = []
        postag_vectors=[]

        for sentence in X:
            for word in sentence[1]:
                word_sizes.append(word[0])
                word_tensors.append(word[1])

            for bert_vector in sentence[2]:
                bert_vectors.append(torch.tensor(bert_vector))
                
            for postag in sentence[3]:
                postag_vectors.append(torch.tensor(postag))

        word_tensors = pad_sequence(word_tensors, batch_first=True)
        word_tensors.to(DEVICE)
        word_sizes = torch.tensor(word_sizes).to(DEVICE)
        word_tensors_packed = pack_padded_sequence(
            word_tensors, word_sizes, batch_first=True, enforce_sorted=False)
        word_tensors_packed.to(DEVICE)

        hidden = self.__init_secondary_hidden(len(word_sizes))
        output, _ = self.word_lstm(word_tensors_packed, hidden)
        output, _ = pad_packed_sequence(output, batch_first=True)

        word_representation = self.__get_significant_lstm_output(
            output, word_sizes)
        bert_embeddings = torch.stack(bert_vectors).squeeze(1)
        postags=torch.stack(postag_vectors).squeeze(1)
        word_vectors = torch.cat((word_representation, bert_embeddings, postags), dim=1)
        word_vectors.to(DEVICE)

        sentences_tensors = []
        count = 0
        for size in sentences_sizes:
            sentences_tensors.append(word_vectors[0: size])
            count += size

        sentences_vectors = pad_sequence(sentences_tensors, batch_first=True)
        sentences_vectors_packed = pack_padded_sequence(
            sentences_vectors, sentences_sizes, batch_first=True, enforce_sorted=False)

        hidden = self.__init_main_hidden(batch_size)
        output, _ = self.sentence_lstm(sentences_vectors_packed, hidden)

        output, _ = pad_packed_sequence(output, batch_first=True)

        sequence_vectors = self.__get_significants_sequence(output, sentences_sizes)
        words = torch.cat(sequence_vectors)

        tags = self.output_layer(words)

        tags_probs = softmax(tags, 1)

        return tags_probs


    def __get_significants_sequence(self, input, sizes):
        sequences = []
        pad_length = input.shape[1]
        for i, size in enumerate(sizes):
            sequences.append(input[i, :size, :])
        return sequences

    def __get_significant_lstm_output(self, input, lenghts):
        representation_dims = input.shape[-1]
        number_of_words = input.shape[0]
        output = torch.zeros(number_of_words, representation_dims)
        for i, lenght in enumerate(lenghts):
            output[i] = input[i, lenght - 1]
        return output
