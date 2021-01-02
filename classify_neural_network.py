import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import Callback
import random

import statistics 

import traceback
import sys

import efficientnet.tfkeras
from tensorflow.keras.models import load_model

order = []
ratio = 0
overallData = pd.read_csv(r'./features/relabel.csv')

def readData():
    global overallData
    overallData = pd.read_csv(r'./features/relabel.csv')
    overallData = overallData.sample(frac=1)

def createTrainTestSet(df):
    global x_train, x_test, y_train, y_test, order

    order = []

    dimen = df.shape[1] - 1

    correct = 0;
    wrong = 0;

    global ratio

    threshold = 10;
    
    for index, row in df.iterrows():
        if row['type'] < threshold*1000:
            wrong += 1
        else:
            correct += 1

    ratio = correct/wrong
    print(f'ratio: {ratio}|{correct}, {wrong}')
    X = []
    Y = []
    for index, row in df.iterrows():
        point = []
        for column in row:
            point += [column]
        point = point[1:dimen+1]
        if row['type'] < threshold*1000:
            X += [point]
            Y += [[0.0, 1.0]]
            order.append(row['type'])
        else:
            X += [point]
            Y += [[1.0, 0.0]]
            order.append(row['type'])
    print(f'X len: {len(X)}, Y len: {len(Y)}')
    x_train = np.array(X[:int(len(X) * 0.8)])
    x_test = np.array(X[int(len(X) * 0.8):])
    y_train = np.array(Y[:int(len(Y) * 0.8)])
    y_test = np.array(Y[int(len(Y) * 0.8):])
    order = np.array(order[int(len(order)*0.8):])
    
    correct = 0
    wrong = 0
            
    for i in y_train:
        if int(i[0])==1:
            correct += 1
        else:
            wrong += 1
    
    ratio = correct/wrong
    print(f'RRRratio: {ratio}|{correct}, {wrong}')

############### [NEURAL NETWORK CODE] #########################################

#Define early stopping by accuracy
class EarlyStoppingByLossAccuracy(Callback):
    def __init__(self):
        super(Callback, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_Accuracy') is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if logs.get('val_Accuracy') >= 0.95 and logs.get('Accuracy') >= 0.93: #Stop when sufficiently accuracte
            print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True
es = EarlyStoppingByLossAccuracy()

def make_deep_net(input_shape, n_output_classes):
    model = Sequential()
    model.add(Dropout(0.3, input_shape = input_shape))
    model.add(Dense(40, activation = 'tanh', input_shape = input_shape))
    model.add(Dense(2, activation = 'softsign'))
    model.add(Dense(n_output_classes, activation='softmax'))
    model.compile(optimizer="rmsprop", metrics=['categorical_accuracy'], loss='kullback_leibler_divergence', )
    return model

def evaluate_deep_net(model, x_test, y_test):
    score = model.evaluate(x=x_test, y=y_test)
    y_predict = model.predict(x_test)

    confusion = [[0,0],[0,0]]
    wrongtoright = []

    for i in range(len(y_test)):
        actual = int(round(y_test[i][1]))
        predict = int(round(y_predict[i][1]))
        confusion[actual][predict] += 1

        if actual == 1 and predict == 0:
            wrongtoright.append(order[i])
    
    print(confusion)
    return (100*score[1], wrongtoright)

def test_net_structure(deep_net, x_train, y_train, x_test, y_test, batch_size, epochs, wrongweight = 1):
    # Train the Deep Neural Network
    deep_net.fit(x=x_train, y=y_train,
                     validation_data=(x_test, y_test),
                     batch_size=batch_size,
                     epochs=epochs, #More than needed, but should be stopped earlier
                     callbacks=[],
                     verbose=0,
                     class_weight={0 : 1, 1 : ratio * wrongweight}) #Early Stopping
    """
    x_test_correct = []
    y_test_correct = []
    x_test_wrong = []
    y_test_wrong = []
    for i in range(len(x_test)):
        if int(y_test[i][0]) == 1:
            x_test_correct += [list(x_test[i])]
            y_test_correct += [list(y_test[i])]
        else:
            x_test_wrong += [list(x_test[i])]
            y_test_wrong += [list(y_test[i])]
    x_test_correct = np.array(x_test_correct)
    y_test_correct = np.array(y_test_correct)
    x_test_wrong = np.array(x_test_wrong)
    y_test_wrong = np.array(y_test_wrong)
    performanceOnCorrect = evaluate_deep_net(deep_net, x_test=x_test_correct, y_test=y_test_correct)
    performanceOnWrong = evaluate_deep_net(deep_net, x_test=x_test_wrong, y_test=y_test_wrong)
    performanceOnBothA = [(performanceOnCorrect[0] + performanceOnWrong[0]) / 2]
    performanceOnBothB = [performanceOnCorrect[1]+performanceOnWrong[1]]
    performanceOnBoth = performanceOnBothA + performanceOnBothB
    return performanceOnBoth
    """
    return evaluate_deep_net(deep_net, x_test=x_test, y_test=y_test)
# Construct the Deep Neural Network

deep_net_tests = []
deep_net_two_tests = []

def getaccuracy():
    readData()
    createTrainTestSet(overallData)
    deep_net = make_deep_net(input_shape=x_train[0].shape, n_output_classes = 2)
    
    wrongweight = 1 #wrongweight = random.randint(3,20) / 10
    #wrongweight = random.randint(3,20) / 10
    
    result = test_net_structure(deep_net, x_train, y_train, x_test, y_test, batch_size=8, epochs=20, wrongweight=wrongweight)
    with open("result.txt", "a") as f:
        f.write(str(result[0]) + ",")

    return result

def predict(predict, traintest):
    X = []
    dimen = predict.shape[1] - 1

    for index, row in predict.iterrows():
        point = []
        for column in row:
            point += [column]
        point = point[1:dimen+1]
        X += [point]
    x_predict = np.array(X)

    createTrainTestSet(traintest)
    deep_net = make_deep_net(input_shape=x_train[0].shape, n_output_classes = 2)
    print(deep_net.predict(x_predict))
    #print(len(X[0]))

def crosspredict(wrongweight):
    readData()
    half = overallData.shape[0] // 2
    predict(overallData[:half], overallData[half:]) 
    
