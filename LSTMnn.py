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
        self.linear_from_word = nn.Linear(2*hidden_size_words+178, input_size)

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True, bidirectional=True)
        self.linear_to_tag_classes = nn.Linear(hidden_size*2, self.output_size)

        self.softmax = nn.LogSoftmax(dim=1)

    def init_hidden(self, batch_size):
        hidden_state = torch.randn(2, batch_size, self.hidden_size)
        cell_state = torch.randn(2, batch_size, self.hidden_size)
        return (hidden_state, cell_state)

    
    def forward(self, input):
        batch_size = input.shape[0]
        
        input = input.view(-1, 206, 113)
        no_bert= input[:, :28,: ]
        bert=input[:, -178:, :]
        new_bert=torch.empty(bert.shape[0], 178)
        
        for i,word in enumerate(bert):
            zeroes=torch.zeros(1, 113)
            zeroes[0][0]=1
            trans=torch.transpose(zeroes, 0, 1)
            new_bert[i]=torch.matmul(word, trans).squeeze()
            
        
        
        hidden = self.init_hidden(no_bert.shape[0])
        words, hidden = self.lstm_embedding(no_bert, hidden)
        words = words.squeeze()[:, -1, :]
        
        
        
        #new_words=torch.empty(words.shape[0], self.hidden_size*2+178)
        bert_words=torch.cat((new_bert, words), dim=1)
        
        bert_words = self.linear_from_word(bert_words)
        bert_words = bert_words.view(batch_size, -1, 50)
        hidden = self.init_hidden(bert_words.shape[0])
        bert_words, _ = self.lstm(bert_words, hidden)
        # words = words.squeeze()[:, -1, :]
        bert_words = bert_words.reshape(-1, 100)
        bert_words = self.linear_to_tag_classes(bert_words)
        
        tags = self.softmax(bert_words)
        return tags


    # [number_of_sentences, number_of_words, word_feature]
    # word -> [number_of_letters, features_letter]