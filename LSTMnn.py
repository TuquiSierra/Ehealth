import torch
from torch import nn

class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, input_size_words, hidden_size_words, output_size_words):
        super(MyLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size_words = input_size_words
        self.hidden_size_words = hidden_size_words
        self.output_size_words = output_size_words

        self.lstm_embedding = nn.LSTM(input_size=input_size_words, hidden_size=hidden_size_words, batch_first=True, bidirectional=True)
        self.linear_from_word = nn.Linear(2*hidden_size_words, input_size)

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True, bidirectional=True)
        self.linear_to_tag_classes = nn.Linear(hidden_size*2, self.output_size)

        self.softmax = nn.LogSoftmax(dim=1)

    def init_hidden(self, batch_size):
        hidden_state = torch.randn(2, batch_size, self.hidden_size)
        cell_state = torch.randn(2, batch_size, self.hidden_size)
        return (hidden_state, cell_state)

    
    def forward(self, input):
        batch_size = input.shape[0]
        input = input.view(-1, 28, 113)
        hidden = self.init_hidden(input.shape[0])
        words, hidden = self.lstm_embedding(input, hidden)
        words = words.squeeze()[:, -1, :]
        words = self.linear_from_word(words)
        words = words.view(batch_size, -1, 50)
        hidden = self.init_hidden(words.shape[0])
        words, _ = self.lstm(words, hidden)
        # words = words.squeeze()[:, -1, :]
        words = words.reshape(-1, 100)
        words = self.linear_to_tag_classes(words)
        
        tags = self.softmax(words)
        return tags


    # [number_of_sentences, number_of_words, word_feature]
    # word -> [number_of_letters, features_letter]