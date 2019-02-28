from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Input
from keras.models import Sequential
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def split_test_train(data, prop=.8):
    split_index = round(len(data)*prop)
    train = data[:split_index]
    test = data[split_index:]
    return train, test

data = pd.read_csv("data/bbc-text.csv")
train_x, test_x = split_test_train(data["text"])
train_y, test_y = split_test_train(data["category"])

print("Training set size: {0}, Test set size: {1}".format(len(train_x), len(test_x)))

#Preprocess X
max_words = 1000
t = Tokenizer(num_words = max_words)
t.fit_on_texts(train_x)
train_one_hot_x = t.texts_to_matrix(train_x, mode = 'count')
test_one_hot_x = t.texts_to_matrix(test_x, mode = 'count')

#Preprocess Y
values = list(set(train_y))

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(train_y)
y_encoded = y_encoded.reshape(len(y_encoded), 1)
onehot_encoder = OneHotEncoder(sparse=False)
train_y_onehot= onehot_encoder.fit_transform(y_encoded)

y_encoded = label_encoder.fit_transform(test_y)
y_encoded = y_encoded.reshape(len(y_encoded), 1)
onehot_encoder = OneHotEncoder(sparse=False)
test_y_onehot= onehot_encoder.fit_transform(y_encoded)

#Define model
model = Sequential()
model.add(Dense(64,
            activation = "relu",
            input_shape=(max_words, )))
model.add(Dense(len(values), activation = "softmax"))
print(model.summary())

model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_one_hot_x, train_y_onehot,
          batch_size=10,
          epochs=25,
          shuffle=True,
          validation_data=(test_one_hot_x, test_y_onehot))

print(model.evaluate(test_one_hot_x, test_y_onehot,
                       batch_size=200, verbose=1))
model.save("BBC.h5")
print("Training complete :)")
