
# ========================Load data=========================

import numpy as np

import pandas as pd

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



from keras.layers import Input, Embedding, Activation, Flatten, Dense

from keras.layers import Conv1D, MaxPooling1D, Dropout

from keras.models import Model



train_data_source = './data/train.csv'

test_data_source = './data/test.csv'



train_df = pd.read_csv(train_data_source, header=None)

test_df = pd.read_csv(test_data_source, header=None)



# concatenate column 1 and column 2 as one text

for df in [train_df, test_df]:

    df[1] = df[1] + df[2]

    df = df.drop([2], axis=1)



# convert string to lower case

train_texts = train_df[1].values

train_texts = [s.lower() for s in train_texts]



test_texts = test_df[1].values

test_texts = [s.lower() for s in test_texts]



# =======================Convert string to index================

# Tokenizer

tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')

tk.fit_on_texts(train_texts)

# If we already have a character list, then replace the tk.word_index

# If not, just skip below part



# -----------------------Skip part start--------------------------

# construct a new vocabulary

alphabet = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"

char_dict = {}

for i, char in enumerate(alphabet):

    char_dict[char] = i + 1



# Use char_dict to replace the tk.word_index

tk.word_index = char_dict.copy()

# Add 'UNK' to the vocabulary

tk.word_index[tk.oov_token] = max(char_dict.values()) + 1

# -----------------------Skip part end----------------------------



# Convert string to index

train_sequences = tk.texts_to_sequences(train_texts)

test_texts = tk.texts_to_sequences(test_texts)



# Padding

train_data = pad_sequences(train_sequences, maxlen=1014, padding='post')

test_data = pad_sequences(test_texts, maxlen=1014, padding='post')



# Convert to numpy array

train_data = np.array(train_data, dtype='float32')

test_data = np.array(test_data, dtype='float32')



# =======================Get classes================

train_classes = train_df[0].values[1:]

train_class_list = [int(x) - 1 for x in train_classes]



test_classes = test_df[0].values[1:]

test_class_list = [int(x) - 1 for x in test_classes]



from keras.utils.np_utils import to_categorical



train_classes = to_categorical(train_class_list)

test_classes = to_categorical(test_class_list)


vocab_size=len(tk.word_index)

embedding_weights=[]
embedding_weights.append(np.zeros(vocab_size))

for char, i in tk.word_index.items():
    onehot=np.zeros(vocab_size)
    onehot[i-1]=1
    embedding_weights.append(onehot)
embedding_weights=np.array(embedding_weights)

input_size=1014
embedding_size=69
conv_layers=[[256, 7, 3], [256, 7, 3], [256, 3 ,-1], [256, 3, -1], [256, 3, -1], [256, 3, 3]]
fully_connected_layers=[1024, 1024]
num_of_classes=4
dropout_p=0.5
optimizer='adam'
loss='categorical_crossentropy'



embedding_layer=Embedding(vocab_size+1, embedding_size, input_length=input_size, weights=[embedding_weights])

inputs=Input(shape=(input_size, ), name='input', dtype='int64')

x=embedding_layer(inputs)

for filter_num, filter_size, pooling_size in conv_layers:
    x=Conv1D(filter_num, filter_size)(x)
    x=Activation('relu')(x)
    if pooling_size!=-1:
        x=MaxPooling1D(pool_size=pooling_size)(x)
x=Flatten()(x)

for dense_size in fully_connected_layers:
    x=Dense(dense_size, activation='relu')(x)
    x=Dropout(dropout_p)(x)

predictions=Dense(num_of_classes, activation='softmax')(x)


model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
model.summary()

indices = np.arange(train_data.shape[0]-1)

np.random.shuffle(indices)



x_train = train_data[indices]

y_train = train_classes[indices]



x_test = test_data

y_test = test_classes



# Training

model.fit(x_train, y_train,

          validation_data=(x_test, y_test),

          batch_size=10,

          epochs=10,

          verbose=2)
    