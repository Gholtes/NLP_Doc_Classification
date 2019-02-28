from loadWordVectors import textToMatrix, load
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Input, Conv1D, Flatten, MaxPooling1D
from keras.models import Sequential
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import time

def split_test_train(data, prop=.8):
    split_index = round(len(data)*prop)
    train = data[:split_index]
    test = data[split_index:]
    return train, test

data = pd.read_csv("data/bbc-text.csv")
train_x, test_x = split_test_train(data["text"])
train_y, test_y = split_test_train(data["category"])

print("Training set size: {0}, Test set size: {1}".format(len(train_x), len(test_x)))

preprocess_start = time.time()
#Preprocess X
maxWords = 400
wordVectors = load('gensimGlove/Glove100d.txt')
# wordVectors = {"the":[1,2,3,4,5], "unk":[0,0,0,1,0]}
wordDims = len(wordVectors["the"])
trainX_Matrix = textToMatrix(train_x, maxWords, wordVectors)
testX_Matrix = textToMatrix(test_x, maxWords, wordVectors)

# print(trainX_Matrix)

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

preprocess_end = time.time()

#Define model
filterSize1 = 5
filterSize2 = 10

model = Sequential()

#First Filter
model.add(Conv1D(filters = 512,
                kernel_size = filterSize1,
                    input_shape=(maxWords, wordDims),
                    padding = 'valid',
                    strides = 1,
                    activation = 'relu',
                    use_bias = True))
model.add(MaxPooling1D(pool_size = maxWords + 1 - filterSize1,
                        strides = 2,
                        padding = 'valid'))
model.add(Flatten())
model.add(Dense(len(values), activation = "softmax"))
print(model.summary())

model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

train_start = time.time()
model.fit(trainX_Matrix, train_y_onehot,
          batch_size=10,
          epochs=25,
          shuffle=True,
          validation_data=(testX_Matrix, test_y_onehot))
train_end = time.time()

eval_start = time.time()
print(model.evaluate(testX_Matrix, test_y_onehot,
                       batch_size=200, verbose=1))
eval_end = time.time()
model.save("BBC.h5")
print("Training complete :)")

print("Preprocess time: {0}".format(preprocess_end-preprocess_start))
print("Training time: {0}".format(train_end-train_start))
print("Eval time: {0}".format(eval_end-eval_start))
