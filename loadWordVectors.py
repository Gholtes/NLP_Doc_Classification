import gensim
import numpy as np
'''use python 3+'''

def convert(path, outputPath):
    #convert GLOVE to gensim word2vec
    gensim.scripts.glove2word2vec.glove2word2vec(path, outputPath)
    print("Word Vectors converted")

def load(path):
    #load word vectors
    wordVectors = gensim.models.KeyedVectors.load_word2vec_format(path, binary=False)
    print("Word Vectors Imported")
    return wordVectors

def textToMatrix(x, maxWords, wordVectors):
    wordDims = len(wordVectors["the"])
    DataPoints = len(x)
    xArray = np.empty((DataPoints, maxWords, wordDims))
    arrayIndex = 0
    for text in x:
        wordList = text.split()
        # print(len(wordList))
        for wordIndex in range(maxWords):
            try:
                #Try get word vec
                xArray[arrayIndex, wordIndex, :] = np.array(wordVectors[wordList[wordIndex]])
            except KeyError:
                #Replace with unknown word vector: key = 'unk'
                xArray[arrayIndex, wordIndex, :] = np.array(wordVectors["unk"])
            except IndexError:
                #Zero Pad
                xArray[arrayIndex, wordIndex, :] = np.zeros((wordDims))
        arrayIndex += 1
    return xArray
