import pandas as pd
import numpy as np
import scipy.stats as stats
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# load data
data = pd.read_csv('dataset-HAR-PUC-Rio.csv', delimiter=';', decimal=',')

# replace categorical with numerical values
data['gender'] = data['gender'].replace({'Woman': 0, 'Man': 1})
data['class'] = data['class'].replace(['sitting', 'sittingdown', 'standing', 'standingup', 'walking'], [0, 1, 2, 3, 4])

# select input features
input_features = ['age', 'how_tall_in_meters', 'weight', 'body_mass_index', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'x3', 'y3', 'z3', 'x4', 'y4', 'z4']

# normalize input data with Min-Max scaler
scaler = MinMaxScaler(feature_range=(0,1))
data[input_features] = scaler.fit_transform(data[input_features])

# shuffle the data
data = data.sample(frac=1).reset_index(drop=True)

# print dataset description
print(data.describe())

# define number of folds
n_folds = 5

# define neural network architecture
def create_model():
    model = Sequential()
    model.add(Dense(17, input_dim=len(input_features), activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['mse', 'accuracy'])
    return model

# create stratified KFold
kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# train and evaluate model for each fold
for fold, (train_index, test_index) in enumerate(kf.split(data, data['class'])):
    print(f'Fold {fold+1}')
    X_train, X_test = data.iloc[train_index][input_features], data.iloc[test_index][input_features]
    y_train, y_test = pd.get_dummies(data.iloc[train_index]['class']).values, pd.get_dummies(data.iloc[test_index]['class']).values

    # create model
    model = create_model()

    # set up early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')

    # train model with early stopping and a validation set
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stop])

    # evaluate model on the test set
    loss, mse, accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f'Accuracy: {accuracy}, MSE: {mse}')

import matplotlib.pyplot as plt
import numpy as np

def plot_average_metrics(histories):
    all_metrics = set()
    for history in histories:
        all_metrics.update(history.history.keys())
    if 'epoch' in all_metrics:
        all_metrics.remove('epoch')
    
    average_metrics = {metric: [] for metric in all_metrics}

    num_epochs = np.min([len(history.history['loss']) for history in histories])
    for epoch in range(num_epochs):
        epoch_metrics = {metric: [] for metric in all_metrics}
        for history in histories:
            for metric in all_metrics:
                epoch_metrics[metric].append(history.history[metric][epoch])
        for metric in all_metrics:
            avg_metric = sum(epoch_metrics[metric]) / len(epoch_metrics[metric])
            average_metrics[metric].append(avg_metric)
    
    for metric in average_metrics:
        plt.plot(average_metrics[metric], label=metric)

    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.xticks(range(num_epochs))
    plt.yticks([i/10 for i in range(21)])
    plt.xlim([0, num_epochs-1])
    plt.ylim([0, 1])
    plt.legend()
    plt.show()

plot_average_metrics(list_of_histories)
print(np.mean(list_of_accuracies, axis=0))
