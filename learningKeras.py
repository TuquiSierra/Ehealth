import numpy as np
import matplotlib.pyplot as plt
from keras.models import Input, Model
from keras.layers import Dense


if __name__ == '__main__':
    n_row = 1000
    x1 = np.random.randn(n_row)
    x2 = np.random.randn(n_row)
    x3 = np.random.randn(n_row)
    y_classifier = np.array([1 if (x1[i] + x2[i] + x3[i]/3 + np.random.randn(1) > 1) else 0 for i in range(n_row)])
    y_cts = x1 + x2 + x3/3 + np.random.randn(n_row)
    data = np.array([x1, x2, x3]).transpose()

    # plt.scatter(data[:,0], data[:,1], c=y_cts)
    # plt.show()

    idx_list = np.linspace(0,999,num=1000)
    idx_test = np.random.choice(n_row, size = 200, replace=False)
    idx_train = np.delete(idx_list, idx_test).astype('int')

    dat_train = data[idx_train,:]
    dat_test = data[idx_test,:]
    y_classifier_train = y_classifier[idx_train]
    y_classifier_test = y_classifier[idx_test]
    y_cts_train = y_cts[idx_train]
    y_cts_test = y_cts[idx_test]
    

    inputs = Input(shape=(3, ))
    outputs = Dense(1, activation='sigmoid')(inputs)
    logistic_model = Model(inputs, outputs)

    logistic_model.compile(optimizer="sgd", loss="binary_crossentropy", metrics=["accuracy"])

    logistic_model.optimizer.lr = 0.001
    logistic_model.fit(x=dat_train, y = y_classifier_train, epochs=5, validation_data= (dat_test, y_classifier_test), verbose=1)