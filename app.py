
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

#train_df = pd.read_csv('csv/testing1.csv')
train_df = pd.read_csv('df_train.csv')
print(train_df.head())

# shuffling data
random_seed = 10
train_df = shuffle(train_df, random_state=random_seed)

# creating new dataframe by removing target 
train_df_X = train_df.drop(columns = ['Unnamed: 0', 'category', 'timestamp'])
print(train_df_X.head())

# creating target category column
# one hot encoding category column
labelencoder_df = LabelEncoder()
train_df['category'] = labelencoder_df.fit_transform(train_df['category'])
#print(train_df['category'])
train_df_y = to_categorical(train_df['category'])
print(train_df_y[0:5])

#def dl_model_multiClassifier(train_df_X, train_df_y)
# create model
model = Sequential()

n_cols = train_df_X.shape[1]

# add layers to model 
model.add(Dense(250, activation = 'relu', input_shape = (n_cols,)))
model.add(Dense(250, activation = 'relu'))
model.add(Dense(250, activation = 'relu'))
model.add(Dense(250, activation = 'relu'))
model.add(Dense(250, activation = 'relu'))
model.add(Dense(250, activation = 'relu'))
model.add(Dense(250, activation = 'relu'))
model.add(Dense(250, activation = 'relu'))
model.add(Dense(4, activation = 'softmax'))

# compile model using accuracy to measure model performance
model.compile(optimizer = 'adam',
  loss = 'categorical_crossentropy',
  metrics = ['accuracy'])

# set early stopping monitor
early_stopping_monitor = EarlyStopping(patience = 3) 

# train model 
model.fit(train_df_X,
 train_df_y, 
 epochs = 30, 
 validation_split = 0.2, 
 callbacks = [early_stopping_monitor])

# test model
test_df = pd.read_csv('csv/test/test2.csv')
test_df = test_df.dropna()
#print(test_df.head())

# shuffling data
random_seed = 10
test_df = shuffle(test_df, random_state=random_seed)

test_df_X = test_df.drop(columns = ['Unnamed: 0', 'category', 'timestamp'])
#print(test_df_X.head())

test_df_pred = model.predict(test_df_X)
#print(test_df_pred)

predictions = []
for x in test_df_pred:
    predictions.append(np.argmax(x) + 1)

test_df['pred_category'] = predictions
print(test_df.head())

#labelencoder_df_test = LabelEncoder()
#test_df['category'] = labelencoder_df.fit_transform(test_df['category'])
#print(test_df['category'])

mislabel = np.sum((test_df['category'] != test_df['pred_category']))
print("Total number of mislabelled data points from {} test samples is = {}".format(len(test_df), mislabel))
accuracy = (len(test_df) - mislabel) / len(test_df) * 100
print(accuracy)