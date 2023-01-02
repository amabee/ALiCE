import nltk
# """
#     The function chat() is a function that takes no arguments and returns nothing. It prints a message
#     to the user, then enters a while loop. The while loop will continue until the user types "quit". The
#     user's input is stored in the variable inp. If the user types "quit", the loop will break.
#     Otherwise, the program will run the user's input through the model.predict() function, which will
#     return a list of probabilities. The program will then find the index of the highest probability in
#     the list, and use that index to find the corresponding tag in the labels list. The program will then
#     search the intents.json file for a tag that matches the one it found, and print a random response
#     from the tag's list of responses
        
#     :param s: The sentence to be classified
#     :param words: The bag of words model
#     :return: The chat function is returning the random choice of responses.
#     """
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
from tensorflow.python.framework import ops
import random
import json
import pickle
# Pickle is a module that converts a python object (list, dict, etc.) into a character stream.
# The idea is that this character stream contains all the information necessary to reconstruct
# the object in another python script.


with open("intents.json") as file:
    data = json.load(file)

try:
    with open ("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])


            if intent["tag"] not in labels:
                labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    output_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        word = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in word:
                bag.append(1)
            else:
                bag.append(0)

        output_row = output_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle" , "wb") as f:
        pickle.dump((words, labels, training, output),f)


ops.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation = "softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    
    return numpy.array(bag)

def chat():
    print("Start Talking With ALiCE! (type 'quit' to stop)")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print("ALiCE:" + random.choice(responses))


chat()