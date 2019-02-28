from keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
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
max_words = 1000
t = Tokenizer(num_words = max_words)
t.fit_on_texts(train_x)
train_one_hot_x = t.texts_to_matrix(train_x)#, mode = 'count')
test_one_hot_x = t.texts_to_matrix(test_x)#, mode = 'count')

#Preprocess Y
values = list(set(train_y))

label_encoder = LabelEncoder()
y_encoded_train = label_encoder.fit_transform(train_y)
y_encoded_test = label_encoder.fit_transform(test_y)

preprocess_end = time.time()

#Define model
model = LogisticRegression(solver='lbfgs',
                            multi_class='multinomial',
                            max_iter = 2000
                            )
train_start = time.time()
model.fit(train_one_hot_x, y_encoded_train)
train_end = time.time()

eval_start = time.time()
print("Logistic Model Accuracy: {0}%".format(round(100*model.score(test_one_hot_x, y_encoded_test),4)))
eval_end = time.time()

print("Preprocess time: {0}".format(preprocess_end-preprocess_start))
print("Training time: {0}".format(train_end-train_start))
print("Eval time: {0}".format(eval_end-eval_start))
