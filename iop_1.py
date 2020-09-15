from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM

def split_dataset(data):
    train,test = data[1:-328], data[-328:-6]
    train = array(split(train, len(train)/7))
    test = array(split(test, len(test)/7))
    return train, test

def evaluate_forecasts(actual, predicted):
    scores = list()
    for i in range(actual.shape[1]):
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        rmse = sqrt(mse)
        scores.append(rmse)
    s=0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row,col] - predicted[row,col])**2
    score = sqrt(s/(actual.shape[0]*actual.shape[1]))
    return score,  scores

def summarize_scores(name, score, scores):
    s_scores = ', '. join(['%1f' % s for s in scores])
    print('%s: [%.3f] %s' %(name, score, scores))

def to_supervised(train, n_input, n_out=7):
    data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
    X, Y= list(), list()
    in_start = 0
    for _ in range(len(data)):
        in_end = in_start + n_input
        out_end = in_end + n_out
        if out_end <= len(data):
            x_input = data[in_start: in_end, 0]
            x_input = x_input.reshape(len(x_input),1)
            X.append(x_input)
            Y.append(data[in_end:out_end, 0])
        in_start += 1
    return array(X), array(Y)

def build_model(train, n_input):
    train_x, train_y = to_supervised(train, n_input)
    verbose, epochs, batch_size = 0, 70, 16
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape = (n_timesteps, n_features)))
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(n_outputs))
    model.compile(loss = 'mse', optimizer='adam')
    model.summary()
    model.fit(train_x, train_y, epochs = epochs, batch_size = batch_size, verbose = verbose)
    return model

def forecast(model, history, n_input):
    data = array(history)
    data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
    input_x = data[-n_input:, 0]
    input_x = input_x.reshape((1, len(input_x), 1))
    yhat = model.predict(input_x, verbose=0)
    yhat = yhat[0]
    print(yhat)
    return yhat

def evaluate_model(train, test, n_input):
    model = build_model(train, n_input)
    history = [x for x in train]
    predictions = list()
    for i in range(len(test)):
        yhat_sequence = forecast(model, history, n_input)
        predictions.append(yhat_sequence)
        history.append(test[i, :])
    predictions = array(predictions)
    score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    return score, scores

dataset = read_csv('household_power_consumption_days.csv', header = 0, infer_datetime_format=True, parse_dates=['datetime'], index_col = ['datetime'])
train, test = split_dataset(dataset.values)
n_input = 7
score, scores = evaluate_model(train, test, n_input)
summarize_scores('LSTM', score, scores)
days = ['sun', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat']
pyplot.plot(days, scores, marker = 'o', label = 'lstm')
pyplot.xlabel("Days")
pyplot.ylabel("RMSE(kW)")
pyplot.show()   